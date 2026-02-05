# FAQ Part 1: The Basics

## Q: What is a semi-CRF?

A Semi-Markov Conditional Random Field is a model that predicts **segments** (contiguous regions with a label and a duration), not individual positions. Where a standard classifier says "position 47 is an exon," a semi-CRF says "positions 40 through 55 are one exon segment, and the next segment is an intron from 56 to 200."

The "semi-Markov" part means the model explicitly reasons about how long each segment lasts, unlike a standard linear-chain CRF which only predicts one label per position. This is important whenever your data has structure: genes have exons and introns with characteristic lengths, ECG traces have heartbeat phases of expected duration, and speech has phonemes that last for varying amounts of time.

## Q: What does torch-semimarkov actually compute?

At its core, the library solves a dynamic programming problem: given a set of scores for every possible segment (label, duration, transition), it efficiently computes quantities like:

- **Partition function** — the normalizing constant for the probability distribution over all valid segmentations
- **Best segmentation** — Viterbi decoding
- **Marginal probabilities** — how likely is a boundary at each position?
- **Entropy** — how uncertain is the model overall?

These are the same computations that underpin CRF-based NER taggers, gene finders, and speech recognizers. The difference is scale: torch-semimarkov is designed to handle sequences of 100,000+ positions on a GPU.

## Q: Why not just use a per-position classifier?

A per-position classifier (like a softmax layer on top of a transformer) predicts each position independently. This means it cannot enforce structural constraints: it might predict an exon at position 50 and an intron at position 51 with no awareness that exons are typically hundreds of bases long. A semi-CRF treats the entire segmentation as a structured object and finds the globally best (or globally probable) assignment, respecting duration and transition constraints.

In practice, this means fewer fragmented predictions, more biologically coherent output, and principled uncertainty estimates about where boundaries fall.

## Q: When should I NOT use a semi-CRF?

If your task doesn't have meaningful segment structure (e.g., per-token sentiment classification), a regular CRF or even a simple softmax may suffice. Semi-CRFs add value specifically when **segment duration matters** and when you want the model to reason about the length and transitions between contiguous regions.

## Q: What is the partition function and why do I keep seeing it?

The partition function $Z$ is the sum of scores over *all possible segmentations* of a sequence. You need it for two things:

1. **Turning raw scores into probabilities** — dividing by $Z$ normalizes the distribution
2. **Computing the training loss** — the negative log-likelihood is $\log Z$ minus the score of the correct segmentation

Computing $\log Z$ efficiently is the core algorithmic challenge, and it's what most of the library's code is devoted to.

---

**See also:** [Using the Library](02-using-the-library.md) · [Jargon Decoder](03-jargon-decoder.md) · [Semirings guide](../semirings.md)
