# Integration with upstream encoders

This guide shows how to use `torch-semimarkov` as a structured prediction layer
on top of upstream encoders like transformers, Mamba SSMs, CNNs, or any
sequence model.

## Conceptual model

```
Input sequence (batch, T, input_dim)
        │
        ▼
┌────────────────────┐
│      Encoder       │   BERT, Mamba, CNN, BiLSTM, etc.
│ (any architecture) │
└────────────────────┘
        │
        ▼
Hidden states (batch, T, hidden_dim)
        │
        ▼
┌───────────────────┐
│  Projection head  │   Linear layers to semi-CRF parameters
└───────────────────┘
        │
        ▼
Edge potentials (batch, T-1, K, C, C)  -or-  HSMM params
        │
        ▼
┌───────────────────┐
│    SemiMarkov     │   Structured inference (this library)
└───────────────────┘
        │
        ▼
Log partition / Viterbi path / Marginals
```

The encoder is **your choice**. This library handles the final structured
prediction layer.

> **Note**: The code examples in this guide are simplified illustrations to
> demonstrate the integration pattern, not production-ready implementations.
> Parameter choices (hidden dimensions, layer counts, K, C values) are
> placeholders and you should tune these for your specific task and data.

## Two parameterization approaches

### 1. Direct edge potentials

Predict the full `(batch, T-1, K, C, C)` tensor directly from encoder outputs.
Each position gets its own learned potentials for all duration and label
combinations.

**What the dimensions mean:**
- `batch`: Independent sequences
- `T-1`: Each position n (from 0 to T-2) predicts segments that *end* at n+1
- `K`: Possible segment durations (1 to K)
- `C, C`: Transition from previous label (last dim) to current label (second-to-last)

So `edge[b, n, k, c2, c1]` is the score for: "at position n in sequence b,
a segment of duration k with label c2 ends here, and the previous segment
had label c1."

**When to use direct edge potentials:**
- When transition patterns vary by position (e.g., near sequence boundaries)
- When you have enough data to learn position-specific patterns
- When the HSMM factorization is too restrictive for your domain
- When you want the encoder to learn everything end-to-end

```python
class DirectEdgeHead(nn.Module):
    def __init__(self, hidden_dim, K, C):
        super().__init__()
        self.K = K
        self.C = C
        # Project to edge potentials
        self.edge_proj = nn.Linear(hidden_dim, K * C * C)

    def forward(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        batch, T, _ = hidden_states.shape
        # Use positions 0..T-2 to predict edges
        edge_hidden = hidden_states[:, :-1, :]  # (batch, T-1, hidden_dim)
        edge_flat = self.edge_proj(edge_hidden)  # (batch, T-1, K*C*C)
        return edge_flat.view(batch, T - 1, self.K, self.C, self.C)
```

**Variant: Context window for edge prediction**

Instead of predicting edges from a single position, you can use a window of
context around each position:

```python
class ContextEdgeHead(nn.Module):
    """Predict edge potentials using local context window."""

    def __init__(self, hidden_dim, K, C, context_size=5):
        super().__init__()
        self.K = K
        self.C = C
        self.context_size = context_size
        # MLP over concatenated context
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * context_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K * C * C)
        )

    def forward(self, hidden_states):
        batch, T, H = hidden_states.shape
        pad = self.context_size // 2

        # Pad and unfold to get context windows
        padded = F.pad(hidden_states, (0, 0, pad, pad))  # (batch, T+2*pad, H)
        windows = padded.unfold(1, self.context_size, 1)  # (batch, T, H, ctx)
        windows = windows.reshape(batch, T, -1)  # (batch, T, H*ctx)

        # Predict edges from context (positions 0..T-2)
        edge_flat = self.edge_mlp(windows[:, :-1])
        return edge_flat.view(batch, T - 1, self.K, self.C, self.C)
```

**Inspecting learned edge potentials**

With direct edges, you can visualize what the model predicts at specific
positions to understand position-dependent patterns:

```python
# After a forward pass, examine edges at specific positions
edge = model(x, lengths)[1]  # (batch, T-1, K, C, C)

# Average edge potentials across a batch for a specific position
pos = 50  # position of interest
avg_edge_at_pos = edge[:, pos].mean(dim=0)  # (K, C, C)

# Plot transition preferences at this position for duration k=1
labels = ["5'UTR", "CDS", "intron", "3'UTR", "intergenic"]
trans_at_pos = avg_edge_at_pos[1]  # duration=1, shape (C, C)

sns.heatmap(trans_at_pos.detach().cpu(), annot=True, fmt=".1f",
            xticklabels=labels, yticklabels=labels, cmap="RdBu_r")
plt.title(f"Learned transition scores at position {pos}")
plt.xlabel("Previous label")
plt.ylabel("Current label")
```

This lets you see if the model learns position-specific behavior, like
different transition patterns near the start vs. middle of sequences.

### 2. HSMM factorization (recommended for most cases)

The Hidden Semi-Markov Model (HSMM) factorization breaks edge potentials into
interpretable components:

- **Initial distribution** `(C,)`: Which segment type is likely to start the sequence?
- **Transitions** `(C, C)`: When a segment ends, what type comes next?
  (e.g., exon → intron is common, exon → exon is rare)
- **Durations** `(C, K)`: How long do segments of each type tend to be?
  (e.g., exons are typically shorter than introns)
- **Emissions** `(batch, T, K, C)`: Given the input at positions n to n+k,
  how well does label c explain the data? This is what your encoder predicts.

This factorization has two practical advantages:

1. **Parameter efficiency**: Instead of learning `T * K * C * C` values,
   you learn `C + C*C + C*K` shared parameters plus `T * K * C` emissions.
   The encoder only needs to predict emissions; transitions and durations
   are learned as small parameter matrices.

2. **Interpretability**: After training, you can extract and visualize the
   learned parameters to understand what the model learned about your domain.

**Example: Inspecting a trained splicing model**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# After training, extract the learned parameters
model = trained_model.head  # your HSMMHead

# Get transition probabilities (C x C matrix)
trans_probs = torch.softmax(model.trans_logits, dim=-1).detach().cpu()

# Get duration probabilities (C x K matrix)
dur_probs = torch.softmax(model.dur_logits, dim=-1).detach().cpu()

# Label names for a splicing model
labels = ["5'UTR", "CDS", "intron", "3'UTR", "intergenic"]

# Plot transition matrix as heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(trans_probs, annot=True, fmt=".2f",
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", ax=axes[0])
axes[0].set_xlabel("Next segment")
axes[0].set_ylabel("Current segment")
axes[0].set_title("Learned transition probabilities")

# Plot duration distributions per label
for i, label in enumerate(labels):
    axes[1].plot(dur_probs[i], label=label)
axes[1].set_xlabel("Duration (positions)")
axes[1].set_ylabel("Probability")
axes[1].set_title("Learned duration distributions")
axes[1].legend()

plt.tight_layout()
plt.savefig("learned_splicing_structure.png")
```

This might reveal, for example:
- CDS → intron and intron → CDS have high probability (expected for splicing)
- intron → intron is near zero (introns don't self-transition)
- Introns have a broader duration distribution than exons
- 5'UTR → CDS is common, but CDS → 5'UTR is rare (directional gene structure)

```python
class HSMMHead(nn.Module):
    def __init__(self, hidden_dim, K, C):
        super().__init__()
        self.K = K
        self.C = C

        # Learnable transition parameters (shared across sequences)
        self.init_logits = nn.Parameter(torch.zeros(C))
        self.trans_logits = nn.Parameter(torch.zeros(C, C))
        self.dur_logits = nn.Parameter(torch.zeros(C, K))

        # Per-position emissions from encoder
        self.emission_proj = nn.Linear(hidden_dim, K * C)

    def forward(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        batch, T, _ = hidden_states.shape

        # Emission scores from encoder
        emission_flat = self.emission_proj(hidden_states)  # (batch, T, K*C)
        emission = emission_flat.view(batch, T, self.K, self.C)

        # Normalize transition parameters
        init_z = F.log_softmax(self.init_logits, dim=-1)
        trans_z = F.log_softmax(self.trans_logits, dim=-1)
        dur_z = F.log_softmax(self.dur_logits, dim=-1)

        # Combine into edge potentials
        edge = SemiMarkov.hsmm(init_z, trans_z, dur_z, emission)
        return edge
```

## Integration examples

### With HuggingFace Transformers (BERT, RoBERTa, etc.)

```python
from transformers import AutoModel
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

class BERTSemiCRF(nn.Module):
    def __init__(self, model_name, K, C):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = HSMMHead(self.encoder.config.hidden_size, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, input_ids, attention_mask, lengths):
        # Encode
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, T, hidden_dim)

        # Project to semi-CRF potentials
        edge = self.head(hidden)

        # Compute log partition (for training loss)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge

    def decode(self, input_ids, attention_mask, lengths):
        """Viterbi decoding for inference."""
        from torch_semimarkov.semirings import MaxSemiring

        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        edge = self.head(hidden)

        crf_max = SemiMarkov(MaxSemiring)
        score, _ = crf_max.logpartition(edge, lengths=lengths)
        # Use marginals or backtrack for best path
        return score
```

### With Mamba SSM

```python
from mamba_ssm import Mamba

class MambaSemiCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, C, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x, lengths):
        # x: (batch, T, input_dim) - e.g., one-hot DNA or embeddings
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h) + h  # residual
        h = self.norm(h)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

### With a simple CNN (for genomics)

```python
class CNNSemiCRF(nn.Module):
    """CNN encoder for DNA sequences with semi-CRF decoder."""

    def __init__(self, K, C, hidden_dim=256):
        super().__init__()
        # DNA: 4 channels (A, C, G, T one-hot)
        self.conv = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7),
        )
        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x_onehot, lengths):
        # x_onehot: (batch, T, 4) -> transpose for conv1d
        h = self.conv(x_onehot.transpose(1, 2))  # (batch, hidden, T)
        h = h.transpose(1, 2)  # (batch, T, hidden)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

### With bidirectional LSTM

```python
class BiLSTMSemiCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, C, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x, lengths):
        # Pack for variable length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

## Training

The standard training objective is **conditional log-likelihood**:

```python
def compute_loss(model, x, lengths, gold_edges):
    """
    Negative log-likelihood loss for semi-CRF.

    Args:
        x: Input sequences
        lengths: Sequence lengths
        gold_edges: Ground truth edge indicators (batch, T-1, K, C, C)
                   with 1s marking true segment boundaries
    """
    log_Z, edge = model(x, lengths)

    # Score of gold segmentation: sum of potentials at gold edges
    gold_score = (edge * gold_edges).sum(dim=(1, 2, 3, 4))

    # NLL = log_Z - gold_score
    loss = (log_Z - gold_score).mean()
    return loss
```

For creating `gold_edges` from label sequences, use `SemiMarkov.to_parts()`:

```python
# labels: (batch, T) with -1 for continuation, 0..C-1 for segment starts
gold_edges = SemiMarkov.to_parts(labels, extra=(C, K))
```

## Performance tips

1. **Use the Triton kernel for inference** when on GPU:
   ```python
   from torch_semimarkov.triton_scan import semi_crf_triton_forward
   # Up to 45x faster than PyTorch
   log_Z = semi_crf_triton_forward(edge.cuda(), lengths.cuda())
   ```

2. **Choose K carefully**: Memory and compute scale with K. Use empirical
   quantiles (p95/p99) of segment lengths rather than maximum.

3. **Vectorized scan for training**: When memory permits, use
   `use_vectorized=True` for 2-3x speedup during training.

4. **Batch similar lengths together** to minimize padding waste.

## Memory considerations

The semi-CRF layer adds minimal overhead to your model:

- **Streaming scan** (default): O(KC) memory for the ring buffer, independent
  of sequence length. For typical values (K=100, C=8), this is negligible.
- **Vectorized scan**: O(TKC) memory, scaling with sequence length. Use when
  you have memory headroom and want 2-3x faster training.

In practice, your encoder (transformer, Mamba, etc.) will dominate memory.
The semi-CRF inference layer is lightweight by comparison.

## Common patterns

### Genomics: Long sequences, moderate K and C

```python
# Gene structure annotation
T = 10000      # 10kb chunks
K = 500        # Max exon/intron length in tokens
C = 5          # exon, intron, UTR, intergenic, etc.

# Use streaming scan (O(KC) = O(2500) memory)
model = MambaSemiCRF(input_dim=4, hidden_dim=256, K=K, C=C)
```

### NLP: Shorter sequences, small K

```python
# Named entity recognition with duration
T = 512        # BERT max length
K = 16         # Max entity span
C = 9          # BIO tags or entity types

model = BERTSemiCRF("bert-base-uncased", K=K, C=C)
```

### Time series segmentation

```python
# Activity recognition
T = 1000       # 1000 timesteps
K = 100        # Max activity duration
C = 6          # Activity classes

model = BiLSTMSemiCRF(input_dim=6, hidden_dim=128, K=K, C=C)
```

## Beyond point predictions: Uncertainty at segment boundaries

A key advantage of semi-CRFs over simpler decoders (like per-position softmax)
is that they define a **distribution over segmentations**, not just a single
prediction. This gives you principled uncertainty estimates about where segment
boundaries actually are.

### Why this matters

When you decode with a standard classifier, you get a label per position, but
no sense of how confident the model is about boundary placement. With a
semi-CRF, the model explicitly reasons about segments as units, and you can
ask: "How probable is it that a segment boundary occurs at position n?"

This is valuable for:
- **Knowing when to trust predictions**: High uncertainty regions may need
  manual review or additional data
- **Scientific applications**: In genomics, knowing that an exon-intron
  boundary is uncertain within a 10bp window is more honest than reporting
  a single coordinate
- **Active learning**: Sample regions where boundary uncertainty is high
- **Downstream analysis**: Propagate uncertainty into variant effect
  predictions, isoform quantification, etc.

### Computing marginal probabilities

The `marginals()` method computes posterior probabilities over edges:

```python
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

model = SemiMarkov(LogSemiring)
log_Z, edge_marginals = model.marginals(edge, lengths=lengths)

# edge_marginals has shape (batch, T-1, K, C, C)
# edge_marginals[b, n, k, c2, c1] = P(segment of label c2, duration k,
#                                     ending at position n+1,
#                                     following a segment of label c1)
```

### Visualizing boundary uncertainty

You can aggregate marginals to see where the model is confident vs. uncertain
about segment boundaries:

```python
# Probability of ANY segment boundary at each position
# Sum over all durations, current labels, and previous labels
boundary_prob = edge_marginals.sum(dim=(2, 3, 4))  # (batch, T-1)

# Plot boundary probability along a sequence
plt.figure(figsize=(12, 3))
plt.plot(boundary_prob[0].detach().cpu())
plt.xlabel("Position")
plt.ylabel("P(segment boundary)")
plt.title("Boundary probability along sequence")

# High values = confident boundary; low values = mid-segment
# Intermediate values = uncertain boundary location
```

### Comparing to the Viterbi path

The Viterbi path (via `MaxSemiring`) gives you the single most probable
segmentation, but it discards uncertainty information:

```python
from torch_semimarkov.semirings import MaxSemiring

# Viterbi: single best path (no uncertainty)
model_max = SemiMarkov(MaxSemiring)
best_score, _ = model_max.logpartition(edge, lengths=lengths)

# Marginals: full posterior (preserves uncertainty)
model_log = SemiMarkov(LogSemiring)
log_Z, marginals = model_log.marginals(edge, lengths=lengths)
```

In regions where the posterior is peaked, Viterbi and marginals will agree.
In ambiguous regions, the marginals will show spread across multiple
boundary positions, while Viterbi arbitrarily picks one.

### Entropy as a global uncertainty measure

For a single scalar measure of segmentation uncertainty, you can compute the
entropy of the posterior distribution using `EntropySemiring`:

```python
from torch_semimarkov.semirings import EntropySemiring

model_ent = SemiMarkov(EntropySemiring)
entropy, _ = model_ent.logpartition(edge, lengths=lengths)

# Higher entropy = more uncertainty about the segmentation
# Lower entropy = model is confident in a particular segmentation
```

This is useful for flagging sequences that may need additional review or
where predictions should be treated with caution.
