# Bisection Step 2: Add Adaptive TILE_C

## Overview
This document describes how to add the adaptive TILE_C feature from commit 07456c6 to the baseline (6ba4d02 + 2K beta ring).

## Changes Required

### Change 1: Add _compute_tile_c Function

Add this function just before `launch_streaming_triton_backward` (around line 889):

```python
    def _compute_tile_c(C: int) -> int:
        """Compute optimal TILE_C based on number of classes.

        Uses runtime heuristics to select TILE_C that forces multiple tiles
        at small C, reducing atomic contention. This follows the Flash Attention
        pattern of runtime block size selection (BLOCK_HEADDIM).

        Args:
            C: Actual number of classes (before padding).

        Returns:
            TILE_C value to use for kernel launch.

        Strategy:
            - C_PAD <= 8:  Use TILE_C=4  (forces 2 tiles, reduces hot-spot contention)
            - C_PAD <= 16: Use TILE_C=8  (forces 2 tiles for C in [9, 16])
            - C_PAD > 16:  Use TILE_C=16 (default, matches current behavior)

        Rationale:
            At small C with fixed TILE_C=16, only one tile executes, creating
            maximum atomic contention. By forcing multiple tiles, we achieve
            spatial separation of atomic operations across the C dimension.

        Example:
            >>> _compute_tile_c(4)   # C_PAD=4  -> TILE_C=4  (2 tiles)
            4
            >>> _compute_tile_c(12)  # C_PAD=16 -> TILE_C=8  (2 tiles)
            8
            >>> _compute_tile_c(32)  # C_PAD=32 -> TILE_C=16 (2 tiles)
            16
        """
        C_PAD = _next_power_of_2(C)

        if C_PAD <= 8:
            return 4
        elif C_PAD <= 16:
            return 8
        else:
            return 16
```

**Location**: Insert this function right before line 889 (`def launch_streaming_triton_backward`).

**Note**: The import `from .triton_forward import _next_power_of_2` is already present at line 36, so no import changes needed.

### Change 2: Modify Kernel Launch to Use Adaptive TILE_C

**Current code** (around line 1164):
```python
                TILE_C=16,  # DO NOT INCREASE - TILE_C=32 causes register spills (14x slowdown)
                num_warps=num_warps,
            )
```

**Replace with**:
```python
                TILE_C=tile_c,
                num_warps=num_warps,
            )
```

**And add this line** before the kernel launch (search for "# Launch kernel with device context" or around line 1160):
```python
        # Compute adaptive TILE_C based on number of classes
        # Forces multiple tiles at small C to reduce atomic contention
        # (follows Flash Attention's BLOCK_HEADDIM pattern)
        tile_c = _compute_tile_c(C)

        # Launch kernel with device context for multi-GPU support
        grid = (batch,)
```

### Exact Edit Instructions

**Edit 1**: Insert _compute_tile_c function at line 889

Use the Edit tool to insert the function before the line `def launch_streaming_triton_backward`.

**Edit 2**: Add tile_c computation

Find the line with comment `# Launch kernel with device context` (should be around line 1050-1060 in the current baseline).
Insert these lines before the grid definition:
```python
        # Compute adaptive TILE_C based on number of classes
        # Forces multiple tiles at small C to reduce atomic contention
        # (follows Flash Attention's BLOCK_HEADDIM pattern)
        tile_c = _compute_tile_c(C)

```

**Edit 3**: Change TILE_C=16 to TILE_C=tile_c

At line 1164, change:
```python
                TILE_C=16,  # DO NOT INCREASE - TILE_C=32 causes register spills (14x slowdown)
```
to:
```python
                TILE_C=tile_c,
```

## Testing

After making these changes, run:
```bash
python test_triton_minimal.py
```

**Expected outcomes**:
1. **If test PASSES** (diff < 3e-5): Adaptive TILE_C is not the culprit, proceed to add Mamba clamping
2. **If test FAILS** (diff ~ 0.1-0.4): Adaptive TILE_C is the culprit, diagnose the issue

## Verification

Check that the changes were applied correctly:
```bash
grep -n "_compute_tile_c" src/torch_semimarkov/streaming/triton_backward.py
grep -n "tile_c = _compute_tile_c(C)" src/torch_semimarkov/streaming/triton_backward.py
grep -n "TILE_C=tile_c" src/torch_semimarkov/streaming/triton_backward.py
```

Should show:
- Line ~889: Function definition
- Line ~1160: tile_c computation
- Line ~1164: TILE_C=tile_c usage
