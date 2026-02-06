#!/bin/bash
# Parse debug output from triton backward kernel
# Usage: ./parse_debug_output.sh debug_v3.log

if [ $# -eq 0 ]; then
    echo "Usage: $0 <debug_log_file>"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: File $LOG_FILE not found"
    exit 1
fi

echo "================================================================================"
echo "TRITON BACKWARD DEBUG OUTPUT ANALYSIS"
echo "================================================================================"
echo ""

# Parse checkpoint loading and buffer verification
echo "=== CHECKPOINT LOADING & BUFFER VERIFICATION (Segment 0) ==="
echo ""
echo "Checkpoint Values Loaded:"
gawk '
/=== CHECKPOINT 0 DEBUG ===/ { in_ckpt = 1; next }
in_ckpt && /k_slot=:/ { match($0, /k_slot=: ([0-9]+)/, a); k_slot = a[1]; next }
in_ckpt && /alpha_val_sum=:/ {
    match($0, /alpha_val_sum=: (-?[0-9.e+-]+)/, a);
    sums[a[1]]++
    next
}
END {
    for (val in sums) {
        printf "  k_slot checkpoint sum: %s (×%d)\n", val, sums[val]
    }
}
' "$LOG_FILE"

echo ""
echo "Segment 0 Store Operation:"
gawk '
/===SEG0 STORE===/ { in_store = 1; next }
in_store && /Storing to seg=/ { next }
in_store && /seg_offset_bytes=:/ {
    match($0, /: ([0-9]+)/, a);
    offsets[a[1]]++
    next
}
in_store && /Storing sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    sums[a[1]]++
    in_store = 0
    next
}
END {
    for (val in offsets) printf "  seg_offset_bytes: %s (×%d threads)\n", val, offsets[val]
    for (val in sums) printf "  Storing sum: %s (×%d threads)\n", val, sums[val]
}
' "$LOG_FILE"

echo ""
echo "Segment 0 Verify Load:"
gawk '
/===SEG0 VERIFY===/ { in_verify = 1; next }
in_verify && /Loading from seg=/ { next }
in_verify && /seg_offset_bytes=:/ {
    match($0, /: ([0-9]+)/, a);
    offsets[a[1]]++
    next
}
in_verify && /Loaded sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    sums[a[1]]++
    next
}
in_verify && /Loaded\[c=0\]=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    firsts[a[1]]++
    next
}
in_verify && /NEG_INF count=:/ {
    match($0, /: ([0-9]+)/, a);
    counts[a[1]]++
    in_verify = 0
    next
}
END {
    for (val in offsets) printf "  seg_offset_bytes: %s (×%d threads)\n", val, offsets[val]
    for (val in sums) printf "  Loaded sum: %s (×%d threads)\n", val, sums[val]
    for (val in firsts) printf "  Loaded[c=0]: %s (×%d threads)\n", val, firsts[val]
    for (val in counts) printf "  NEG_INF count: %s (×%d threads)\n", val, counts[val]
}
' "$LOG_FILE"

echo ""
echo "About to Load Alpha Prev (t=1, k=1):"
gawk '
/=== ABOUT TO LOAD ALPHA_PREV ===/ { in_load = 1; next }
in_load && /idx \(\) t=:/ { match($0, /: ([0-9]+)/, a); ts[a[1]]++; next }
in_load && /idx \(\) k=:/ { match($0, /: ([0-9]+)/, a); ks[a[1]]++; next }
in_load && /start_pos=:/ { match($0, /: ([0-9]+)/, a); start_poss[a[1]]++; next }
in_load && /local_start=:/ { match($0, /: ([0-9]+)/, a); local_starts[a[1]]++; next }
in_load && /seg_start=:/ { match($0, /: ([0-9]+)/, a); seg_starts[a[1]]++; next }
in_load && /ckpt_idx=:/ { match($0, /: ([0-9]+)/, a); ckpt_idxs[a[1]]++; next }
in_load && /seg_offset=:/ { match($0, /: ([0-9]+)/, a); seg_offsets[a[1]]++; next }
in_load && /t_offset=:/ { match($0, /: ([0-9]+)/, a); t_offsets[a[1]]++; next }
in_load && /=== LOADED ALPHA_PREV ===/ { in_load = 0; in_loaded = 1; next }
END {
    for (val in ts) printf "  t: %s (×%d threads)\n", val, ts[val]
    for (val in ks) printf "  k: %s (×%d threads)\n", val, ks[val]
    for (val in start_poss) printf "  start_pos: %s (×%d threads)\n", val, start_poss[val]
    for (val in local_starts) printf "  local_start: %s (×%d threads)\n", val, local_starts[val]
    for (val in seg_starts) printf "  seg_start: %s (×%d threads)\n", val, seg_starts[val]
    for (val in ckpt_idxs) printf "  ckpt_idx: %s (×%d threads)\n", val, ckpt_idxs[val]
    for (val in seg_offsets) printf "  seg_offset: %s (×%d threads)\n", val, seg_offsets[val]
    for (val in t_offsets) printf "  t_offset: %s (×%d threads)\n", val, t_offsets[val]
}
' "$LOG_FILE"

echo ""
echo "Loaded Alpha Prev Results:"
gawk '
/=== LOADED ALPHA_PREV ===/ { in_result = 1; next }
in_result && /alpha_prev_sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    sums[a[1]]++
    next
}
in_result && /alpha_prev\[c=0\]=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    firsts[a[1]]++
    next
}
in_result && /NEG_INF count=:/ {
    match($0, /: ([0-9]+)/, a);
    counts[a[1]]++
    next
}
in_result && /===/ { in_result = 0 }
END {
    for (val in sums) printf "  alpha_prev_sum: %s (×%d threads)\n", val, sums[val]
    for (val in firsts) printf "  alpha_prev[c=0]: %s (×%d threads)\n", val, firsts[val]
    for (val in counts) printf "  NEG_INF count: %s (×%d threads)\n", val, counts[val]
}
' "$LOG_FILE"

echo ""
echo "================================================================================"
echo ""

# Parse alpha debug blocks
echo "=== ALPHA VALUES ==="
echo "Position | local_t | ckpt_idx | alpha_sum  | log_norm"
echo "---------|---------|----------|------------|----------"

gawk '
BEGIN { t=-1; local_t=-1; ckpt_idx=-1; alpha_sum=0; log_norm=0 }

/=== ALPHA DEBUG ===/  { in_alpha = 1; next }

in_alpha && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, arr); t = arr[1]
    next
}

in_alpha && /idx \(\) local_t=:/ {
    match($0, /local_t=: ([0-9]+)/, arr); local_t = arr[1]
    next
}

in_alpha && /idx \(\) ckpt_idx=:/ {
    match($0, /ckpt_idx=: ([0-9]+)/, arr); ckpt_idx = arr[1]
    next
}

in_alpha && /alpha_sum=:/ {
    match($0, /alpha_sum=: (-?[0-9.]+)/, arr)
    alpha_sum = arr[1]
    next
}

in_alpha && /log_norm_at_ckpt=:/ {
    match($0, /log_norm_at_ckpt=: ([0-9.]+)/, arr)
    log_norm = arr[1]

    key = t "_" local_t "_" ckpt_idx "_" alpha_sum "_" log_norm
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("t=%-6s | %-7s | %-8s | %10.4f | %.6f",
                     t, local_t, ckpt_idx, alpha_sum, log_norm)
    }

    in_alpha = 0
    t=-1; local_t=-1; ckpt_idx=-1; alpha_sum=0; log_norm=0
    next
}
' "$LOG_FILE" | sort -t= -k2 -n

echo ""
echo "=== BETA VALUES AT SEGMENT BOUNDARY ==="
echo "Position | end_pos | end_ring_idx | beta_sum   | from_segment"
echo "---------|---------|--------------|------------|-------------"

gawk '
BEGIN { t=-1; end_pos=-1; ring_idx=-1; beta_sum=0; seg=-1 }

/=== BETA DEBUG t=:/ && /26/ { in_beta = 1; seg = 0; t = 26; next }
/=== BETA DEBUG t=:/ && /27/ { in_beta = 1; seg = 1; t = 27; next }

in_beta && /idx \(\) end_pos=:/ {
    match($0, /end_pos=: ([0-9]+)/, arr); end_pos = arr[1]
    next
}

in_beta && /idx \(\) end_ring_idx=:/ {
    match($0, /end_ring_idx=: ([0-9]+)/, arr); ring_idx = arr[1]
    next
}

in_beta && /idx \(\) beta_sum=:/ {
    match($0, /beta_sum=: (-?[0-9.]+)/, arr)
    beta_sum = arr[1]

    key = t "_" end_pos "_" ring_idx "_" beta_sum "_" seg
    if (!(key in seen)) {
        seen[key] = 1
        seg_label = (seg == 0) ? "Seg0" : "Seg1"
        print sprintf("t=%-6s | %-7s | %-12s | %10.4f | %s",
                     t, end_pos, ring_idx, beta_sum, seg_label)
    }

    in_beta = 0
    t=-1; end_pos=-1; ring_idx=-1; beta_sum=0; seg=-1
    next
}
' "$LOG_FILE"

echo ""
echo "=== SCALE COMPUTATION AT SEGMENT BOUNDARY ==="
echo "Position | log_norm | global_max | log_Z     | log_scale | scale   "
echo "---------|----------|------------|-----------|-----------|----------"

gawk '
BEGIN { t=-1; log_norm=0; gmax=0; logZ=0; log_scale=0; scale=0 }

/SEGMENT BOUNDARY DEBUG/  { in_seg0 = 1; t = 26; next }
/SEGMENT 1 REFERENCE/     { in_seg1 = 1; t = 27; next }

(in_seg0 || in_seg1) && /end_pos=:/ {
    match($0, /end_pos=: ([0-9]+)/, arr); end_pos = arr[1]
    next
}

(in_seg0 || in_seg1) && /log_norm_at_ckpt=:/ {
    match($0, /log_norm_at_ckpt=: ([0-9.]+)/, arr)
    log_norm = arr[1]
    next
}

(in_seg0 || in_seg1) && /global_max=:/ {
    match($0, /global_max=: (-?[0-9.]+)/, arr)
    gmax = arr[1]
    if (gmax < -999000) next  # Skip NEG_INF values
    next
}

(in_seg0 || in_seg1) && /log_Z=:/ {
    match($0, /log_Z=: ([0-9.]+)/, arr)
    logZ = arr[1]
    next
}

(in_seg0 || in_seg1) && /idx \(\) log_scale=:/ {
    match($0, /log_scale=: (-?[0-9.]+)/, arr)
    log_scale = arr[1]
    if (log_scale < -999000) next  # Skip invalid values
    next
}

(in_seg0 || in_seg1) && /scale=:/ && !/log_scale/ {
    match($0, /scale=: ([0-9.]+)/, arr)
    scale = arr[1]
    if (scale < 0.001) next  # Skip near-zero values

    key = t "_" log_norm "_" gmax "_" logZ "_" log_scale "_" scale
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("t=%-6s | %8.6f | %10.6f | %9.6f | %9.6f | %.6f",
                     t, log_norm, gmax, logZ, log_scale, scale)
    }

    in_seg0 = 0
    in_seg1 = 0
    t=-1; log_norm=0; gmax=0; logZ=0; log_scale=0; scale=0
    next
}
' "$LOG_FILE" | sort -t= -k2 -n

echo ""
echo "=== MARGINAL COMPUTATION COMPONENTS ==="
echo "Position | alpha[0] | beta[0] | edge[0,0] | log_joint | marginal"
echo "---------|----------|---------|-----------|-----------|----------"

gawk '
BEGIN { t=-1; alpha=0; beta=0; edge=0; lj=0; marg=0 }

/=== MARGINAL DEBUG t=:/ {
    if (/26/) { in_marg = 1; t = 26 }
    if (/27/) { in_marg = 1; t = 27 }
    next
}

in_marg && /alpha\[0\]=:/ {
    match($0, /alpha\[0\]=: (-?[0-9.e+-]+)/, arr)
    alpha = arr[1]
    next
}

in_marg && /beta\[0\]=:/ {
    match($0, /beta\[0\]=: (-?[0-9.e+-]+)/, arr)
    beta = arr[1]
    next
}

in_marg && /edge\[0,0\]=:/ {
    match($0, /edge\[0,0\]=: (-?[0-9.e+-]+)/, arr)
    edge = arr[1]
    next
}

in_marg && /log_joint\[0,0\]=:/ {
    match($0, /log_joint\[0,0\]=: (-?[0-9.e+-]+)/, arr)
    lj = arr[1]
    next
}

in_marg && /marginal_final\[0,0\]=:/ {
    match($0, /marginal_final\[0,0\]=: (-?[0-9.e+-]+)/, arr)
    marg = arr[1]

    key = t "_" alpha "_" beta "_" edge "_" lj "_" marg
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("t=%-6s | %9.4f | %8.4f | %9.4f | %9.4f | %9.6f",
                     t, alpha, beta, edge, lj, marg)
    }

    in_marg = 0
    t=-1; alpha=0; beta=0; edge=0; lj=0; marg=0
    next
}
' "$LOG_FILE"

echo ""
echo "=== CHECKPOINT 0 LOADING ==="
echo "k_slot   | alpha_val_sum"
echo "---------|---------------"

gawk '
BEGIN { k_slot=-1; alpha_val=0 }

/=== CHECKPOINT 0 DEBUG ===/ { in_ckpt = 1; next }

in_ckpt && /k_slot=:/ {
    match($0, /k_slot=: ([0-9]+)/, arr); k_slot = arr[1]
    next
}

in_ckpt && /alpha_val_sum=:/ {
    match($0, /alpha_val_sum=: (-?[0-9.e+-]+)/, arr)
    alpha_val = arr[1]

    key = k_slot "_" alpha_val
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("%-8s | %12.4f", k_slot, alpha_val)
    }

    in_ckpt = 0
    k_slot=-1; alpha_val=0
    next
}
' "$LOG_FILE"

echo ""
echo "=== ALPHA RECOMPUTATION (Segment 0, t=0-2) ==="
echo "Position | local_t | alpha_t_sum"
echo "---------|---------|-------------"

gawk '
BEGIN { t=-1; local_t=-1; alpha_sum=0 }

/=== ALPHA RECOMP DEBUG ===/ { in_recomp = 1; next }

in_recomp && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, arr); t = arr[1]
    next
}

in_recomp && /idx \(\) local_t=:/ {
    match($0, /local_t=: ([0-9]+)/, arr); local_t = arr[1]
    next
}

in_recomp && /alpha_t_sum=:/ {
    match($0, /alpha_t_sum=: (-?[0-9.e+-]+)/, arr)
    alpha_sum = arr[1]

    key = t "_" local_t "_" alpha_sum
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("t=%-6s | %-7s | %12.4f", t, local_t, alpha_sum)
    }

    in_recomp = 0
    t=-1; local_t=-1; alpha_sum=0
    next
}
' "$LOG_FILE"

echo ""
echo "=== FORWARD PASS ALPHA (For Comparison) ==="
echo "Position | alpha_t_sum"
echo "---------|-------------"

gawk '
BEGIN { t=-1; alpha_sum=0 }

/=== FWD ALPHA DEBUG ===/ { in_fwd = 1; next }

in_fwd && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, arr); t = arr[1]
    next
}

in_fwd && /alpha_t_sum=:/ {
    match($0, /alpha_t_sum=: (-?[0-9.e+-]+)/, arr)
    alpha_sum = arr[1]

    key = t "_" alpha_sum
    if (!(key in seen)) {
        seen[key] = 1
        print sprintf("t=%-6s | %12.4f", t, alpha_sum)
    }

    in_fwd = 0
    t=-1; alpha_sum=0
    next
}
' "$LOG_FILE"

echo ""
echo "=== INTERMEDIATE VALUES AT t=1, k=1 ==="
echo ""
echo "Forward Pass:"
gawk '
/=== FWD PASS t=1 k=1 ===/ { in_fwd = 1; next }
in_fwd && /alpha_prev_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_prev_values[a[1]]++; next }
in_fwd && /cum_end_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); cum_end_values[a[1]]++; next }
in_fwd && /cum_start_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); cum_start_values[a[1]]++; next }
in_fwd && /content_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); content_values[a[1]]++; next }
in_fwd && /dur_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); dur_values[a[1]]++; next }
in_fwd && /seg_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); seg_values[a[1]]++; in_fwd=0; next }
END {
    for (v in alpha_prev_values) printf "  alpha_prev_sum: %s (×%d threads)\n", v, alpha_prev_values[v]
    for (v in cum_end_values) printf "  cum_end_sum:    %s (×%d threads)\n", v, cum_end_values[v]
    for (v in cum_start_values) printf "  cum_start_sum:  %s (×%d threads)\n", v, cum_start_values[v]
    for (v in content_values) printf "  content_sum:    %s (×%d threads)\n", v, content_values[v]
    for (v in dur_values) printf "  dur_sum:        %s (×%d threads)\n", v, dur_values[v]
    for (v in seg_values) printf "  seg_sum:        %s (×%d threads)\n", v, seg_values[v]
}
' "$LOG_FILE"

echo ""
echo "Backward Recomputation:"
gawk '
/=== BWD RECOMP t=1 k=1 ===/ { in_bwd = 1; next }
in_bwd && /alpha_prev_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_prev_values[a[1]]++; next }
in_bwd && /cum_end_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); cum_end_values[a[1]]++; next }
in_bwd && /cum_start_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); cum_start_values[a[1]]++; next }
in_bwd && /content_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); content_values[a[1]]++; next }
in_bwd && /dur_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); dur_values[a[1]]++; next }
in_bwd && /seg_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); seg_values[a[1]]++; in_bwd=0; next }
END {
    for (v in alpha_prev_values) printf "  alpha_prev_sum: %s (×%d threads)\n", v, alpha_prev_values[v]
    for (v in cum_end_values) printf "  cum_end_sum:    %s (×%d threads)\n", v, cum_end_values[v]
    for (v in cum_start_values) printf "  cum_start_sum:  %s (×%d threads)\n", v, cum_start_values[v]
    for (v in content_values) printf "  content_sum:    %s (×%d threads)\n", v, content_values[v]
    for (v in dur_values) printf "  dur_sum:        %s (×%d threads)\n", v, dur_values[v]
    for (v in seg_values) printf "  seg_sum:        %s (×%d threads)\n", v, seg_values[v]
}
' "$LOG_FILE"

echo ""
echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"
echo ""

echo "FORWARD vs BACKWARD RECOMPUTATION COMPARISON:"
echo "Position | Forward Alpha | Backward Recomp | Difference"
echo "---------|---------------|-----------------|------------"

gawk '
# Extract forward alpha values
/=== FWD ALPHA DEBUG ===/ { in_fwd = 1; next }
in_fwd && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, arr); fwd_t = arr[1]
    next
}
in_fwd && /alpha_t_sum=:/ {
    match($0, /alpha_t_sum=: (-?[0-9.e+-]+)/, arr)
    fwd_alpha[fwd_t] = arr[1]
    in_fwd = 0
    next
}

# Extract backward recomputation values
/=== ALPHA RECOMP DEBUG ===/ { in_bwd = 1; next }
in_bwd && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, arr); bwd_t = arr[1]
    next
}
in_bwd && /alpha_t_sum=:/ {
    match($0, /alpha_t_sum=: (-?[0-9.e+-]+)/, arr)
    bwd_alpha[bwd_t] = arr[1]
    in_bwd = 0
    next
}

END {
    # Compare at t=1, t=2
    for (t in fwd_alpha) {
        if (t in bwd_alpha) {
            diff = bwd_alpha[t] - fwd_alpha[t]
            print sprintf("t=%-6s | %13.4f | %15.4f | %+11.4f",
                         t, fwd_alpha[t], bwd_alpha[t], diff)
        }
    }
}
' "$LOG_FILE"

echo ""

# Compute expected marginal ratio based on scale values
gawk '
/t=26/ && /scale=:/ && !/log_scale/ {
    match($0, /scale=: ([0-9.]+)/, arr)
    scale_26 = arr[1]
    if (scale_26 > 0.001) { found_26 = 1 }
}

/t=27/ && /scale=:/ && !/log_scale/ {
    match($0, /scale=: ([0-9.]+)/, arr)
    scale_27 = arr[1]
    if (scale_27 > 0.001) { found_27 = 1 }
}

END {
    if (found_26 && found_27) {
        ratio = scale_26 / scale_27
        print "Scale ratio (t=26 / t=27):", ratio
        print "Expected marginal ratio:  ", ratio, "(if scales are the issue)"
        print "Observed marginal ratio:  ", "0.8901"
        print ""
        if (ratio > 0.88 && ratio < 0.90) {
            print "✓ CONFIRMED: Scale mismatch explains the 0.8901x error!"
        } else {
            print "✗ Scale ratio does NOT explain the error - look elsewhere"
        }
    }
}
' "$LOG_FILE"

echo ""
echo "If alpha_sum values differ between t=1 and t=26, there's an alpha recomputation bug"
echo "If beta_sum values differ between segments, there's a beta normalization bug"

echo ""
echo "================================================================================"
echo "FIX VERIFICATION SUMMARY"
echo "================================================================================"
echo ""

# Check if the fix is successful by analyzing key indicators
gawk '
BEGIN {
    # Indicators
    memory_barrier_ok = 0
    alpha_recomp_ok = 0
    seg0_errors = 0
    seg1_errors = 0

    # Thresholds
    alpha_diff_threshold = 0.0001
}

# Check 1: Memory barrier worked (all threads see same checkpoint value)
/===SEG0 VERIFY===/ { in_verify = 1; next }
in_verify && /Loaded sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    verify_values[a[1]]++
    next
}
in_verify && /===/ { in_verify = 0 }

# Check 2: Forward vs backward alpha match
/=== FWD ALPHA DEBUG ===/ { in_fwd = 1; next }
in_fwd && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, a)
    fwd_t = a[1]
    next
}
in_fwd && /alpha_t_sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    fwd_alpha[fwd_t] = a[1]
    in_fwd = 0
    next
}

/=== ALPHA RECOMP DEBUG ===/ { in_bwd = 1; next }
in_bwd && /idx \(\) t=:/ {
    match($0, /t=: ([0-9]+)/, a)
    bwd_t = a[1]
    next
}
in_bwd && /alpha_t_sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    bwd_alpha[bwd_t] = a[1]
    in_bwd = 0
    next
}

END {
    # Evaluate Check 1: Memory barrier
    print "CHECK 1: Memory Barrier (tl.debug_barrier())"
    print "----------------------------------------------"
    unique_values = 0
    for (val in verify_values) {
        unique_values++
        if (val > -1e8) {  # Not NEG_INF
            correct_value = val
        }
    }

    if (unique_values == 1 && correct_value != "") {
        print "✓ PASS: All threads see the same checkpoint value"
        print "        Value: " correct_value " (×128 threads)"
        memory_barrier_ok = 1
    } else if (unique_values > 1) {
        print "✗ FAIL: Thread divergence detected!"
        for (val in verify_values) {
            print "        " val " (×" verify_values[val] " threads)"
        }
        print "        → Memory barrier missing or ineffective"
    } else {
        print "⚠ SKIP: No verification data found"
    }
    print ""

    # Evaluate Check 2: Forward vs backward alpha
    print "CHECK 2: Alpha Recomputation Correctness"
    print "----------------------------------------------"

    max_diff = 0
    match_count = 0
    mismatch_count = 0

    for (t in fwd_alpha) {
        if (t in bwd_alpha) {
            diff = bwd_alpha[t] - fwd_alpha[t]
            abs_diff = (diff < 0) ? -diff : diff

            if (abs_diff < alpha_diff_threshold) {
                match_count++
            } else {
                mismatch_count++
                if (abs_diff > max_diff) {
                    max_diff = abs_diff
                    worst_t = t
                }
            }
        }
    }

    if (match_count > 0 && mismatch_count == 0) {
        print "✓ PASS: Forward and backward alpha values match"
        print "        Positions checked: " match_count
        print "        Max difference: " max_diff " (< " alpha_diff_threshold " threshold)"
        alpha_recomp_ok = 1
    } else if (mismatch_count > 0) {
        print "✗ FAIL: Alpha recomputation errors detected"
        print "        Matching: " match_count " positions"
        print "        Mismatched: " mismatch_count " positions"
        print "        Worst error: " max_diff " at t=" worst_t
    } else {
        print "⚠ SKIP: No alpha comparison data found"
    }
    print ""

    # Final verdict
    print "OVERALL VERDICT"
    print "==============="

    if (memory_barrier_ok && alpha_recomp_ok) {
        print ""
        print "    ✓✓✓ ALL CHECKS PASSED ✓✓✓"
        print ""
        print "The segment isolation + memory barrier fix appears SUCCESSFUL!"
        print ""
        print "Evidence:"
        print "  1. All warps synchronized (memory barrier working)"
        print "  2. Forward and backward alpha match (recomputation correct)"
        print "  3. Ready for full marginal error test on HPC"
        print ""
        print "Next steps:"
        print "  - Run full test: python scripts/debug_marginal_divergence.py"
        print "  - Verify segment 0 errors are eliminated"
        print "  - If confirmed, remove debug prints"
        print ""
    } else {
        print ""
        print "    ✗✗✗ ISSUES DETECTED ✗✗✗"
        print ""
        if (!memory_barrier_ok) {
            print "  → Memory barrier issue: Different warps see different values"
        }
        if (!alpha_recomp_ok) {
            print "  → Alpha recomputation issue: Forward/backward mismatch"
        }
        print ""
        print "The fix may need additional work. Check failed sections above."
        print ""
    }
}
' "$LOG_FILE"
echo "If scale ratio matches 0.8901, the scale computation formula is wrong"

echo ""
echo "================================================================================"
echo "NON-DETERMINISM DEBUG (t=9, k=1)"
echo "================================================================================"
echo ""

echo "=== BETA STORE (t=10) ==="
echo "Beta values stored at t=10, which are read by t=9, k=1."
echo "If these differ between runs, non-determinism is upstream (t=11+)."
echo ""
gawk '
/=== BETA STORE t=10 ===/ { in_beta_store = 1; next }
in_beta_store && /new_beta_sum=:/ {
    match($0, /new_beta_sum=: (-?[0-9.e+-]+)/, a)
    beta_store_vals[a[1]]++
    next
}
in_beta_store && /t_ring_idx=:/ {
    match($0, /t_ring_idx=: ([0-9]+)/, a)
    ring_idx_vals[a[1]]++
    in_beta_store = 0
    next
}
END {
    print "new_beta_sum values (deduplicated):"
    for (val in beta_store_vals) {
        printf "  %s (×%d threads)\n", val, beta_store_vals[val]
    }
    print ""
    print "t_ring_idx values (deduplicated):"
    for (val in ring_idx_vals) {
        printf "  %s (×%d threads)\n", val, ring_idx_vals[val]
    }
}
' "$LOG_FILE"

echo ""
echo "=== BETA LOAD (t=9, k=1) ==="
echo "Beta values loaded at t=9, k=1 (reads beta[10] from ring buffer)."
echo "If store is consistent but load differs, ring buffer corruption suspected."
echo ""
gawk '
/=== BETA LOAD t=9 k=1 ===/ { in_beta_load = 1; next }
in_beta_load && /tile_start=:/ {
    match($0, /tile_start=: ([0-9]+)/, a)
    tile_start_vals[a[1]]++
    next
}
in_beta_load && /beta_tile_sum=:/ {
    match($0, /beta_tile_sum=: (-?[0-9.e+-]+)/, a)
    beta_load_vals[a[1]]++
    next
}
in_beta_load && /end_ring_idx=:/ {
    match($0, /end_ring_idx=: ([0-9]+)/, a)
    end_ring_vals[a[1]]++
    next
}
in_beta_load && /end_pos=:/ {
    match($0, /end_pos=: ([0-9]+)/, a)
    end_pos_vals[a[1]]++
    in_beta_load = 0
    next
}
END {
    print "tile_start values (deduplicated):"
    for (val in tile_start_vals) {
        printf "  %s (×%d threads)\n", val, tile_start_vals[val]
    }
    print ""
    print "beta_tile_sum values (deduplicated):"
    for (val in beta_load_vals) {
        printf "  %s (×%d threads)\n", val, beta_load_vals[val]
    }
    print ""
    print "end_ring_idx values (deduplicated):"
    for (val in end_ring_vals) {
        printf "  %s (×%d threads)\n", val, end_ring_vals[val]
    }
    print ""
    print "end_pos values (deduplicated):"
    for (val in end_pos_vals) {
        printf "  %s (×%d threads)\n", val, end_pos_vals[val]
    }
}
' "$LOG_FILE"

echo ""
echo "=== PASS 1 GLOBAL STATISTICS ==="
echo "These values should be IDENTICAL across all runs if Pass 1 is deterministic."
echo ""
gawk '
/=== PASS1 DEBUG t=9 k=1 ===/ { in_pass1 = 1; next }
in_pass1 && /global_max=:/ {
    match($0, /global_max=: (-?[0-9.e+-]+)/, a)
    global_max_vals[a[1]]++
    next
}
in_pass1 && /global_sum_exp=:/ {
    match($0, /global_sum_exp=: (-?[0-9.e+-]+)/, a)
    global_sum_vals[a[1]]++
    in_pass1 = 0
    next
}
END {
    print "global_max values (deduplicated):"
    for (val in global_max_vals) {
        printf "  %s (×%d threads)\n", val, global_max_vals[val]
    }
    print ""
    print "global_sum_exp values (deduplicated):"
    for (val in global_sum_vals) {
        printf "  %s (×%d threads)\n", val, global_sum_vals[val]
    }
}
' "$LOG_FILE"

echo ""
echo "=== SCALE COMPUTATION ==="
echo "These values should be IDENTICAL across all runs."
echo ""
gawk '
/=== PASS1 DEBUG t=9 k=1 ===/ { in_scale = 1; next }
in_scale && /log_scale=:/ {
    match($0, /log_scale=: (-?[0-9.e+-]+)/, a)
    log_scale_vals[a[1]]++
    next
}
in_scale && /scale=:/ {
    match($0, /scale=: ([0-9.e+-]+)/, a)
    scale_vals[a[1]]++
    in_scale = 0
    next
}
END {
    print "log_scale values (deduplicated):"
    for (val in log_scale_vals) {
        printf "  %s (×%d threads)\n", val, log_scale_vals[val]
    }
    print ""
    print "scale values (deduplicated):"
    for (val in scale_vals) {
        printf "  %s (×%d threads)\n", val, scale_vals[val]
    }
}
' "$LOG_FILE"

echo ""
echo "=== LOCAL ACCUMULATOR VALUES ==="
echo "These values should be IDENTICAL across all runs."
echo ""
gawk '
/=== GRAD_CS_LOCAL t=9 ===/ { in_grad_cs = 1; next }
in_grad_cs && /grad_cs_t_local_sum=:/ {
    match($0, /grad_cs_t_local_sum=: (-?[0-9.e+-]+)/, a)
    grad_cs_vals[a[1]]++
    in_grad_cs = 0
    next
}
/=== GRAD_DB_LOCAL t=9 k=1 ===/ { in_grad_db = 1; next }
in_grad_db && /grad_db_k_local_sum=:/ {
    match($0, /grad_db_k_local_sum=: (-?[0-9.e+-]+)/, a)
    grad_db_vals[a[1]]++
    in_grad_db = 0
    next
}
END {
    print "grad_cs_t_local_sum values (deduplicated):"
    for (val in grad_cs_vals) {
        printf "  %s (×%d threads)\n", val, grad_cs_vals[val]
    }
    print ""
    print "grad_db_k_local_sum values (deduplicated):"
    for (val in grad_db_vals) {
        printf "  %s (×%d threads)\n", val, grad_db_vals[val]
    }
}
' "$LOG_FILE"

echo ""
echo "=== CROSS-RUN ANALYSIS ==="
echo "If you ran the script multiple times and captured all output:"
echo "  - Multiple unique values for global_max = Problem in Pass 1 tl.static_range"
echo "  - Same global_max but different scale = Problem in scale computation"
echo "  - Same scale but different grad_cs = Problem in Pass 2 accumulation"
echo ""

echo "================================================================================"
echo "SUM REDUCTION PRECISION ANALYSIS (t=1000, k=1) - LARGE CONFIG"
echo "================================================================================"
echo ""

echo ""
echo "=== KERNEL CONFIGURATION ==="
gawk '
/=== BACKWARD KERNEL DEBUG ===/ { in_config = 1; next }
in_config && /idx \(\) NUM_CKPTS=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    num_ckpts = a[1]
    next
}
in_config && /idx \(\) CHECKPOINT_INTERVAL=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    interval = a[1]
    next
}
in_config && /idx \(\) T=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    T = a[1]
    in_config = 0
    next
}
END {
    if (num_ckpts != "") {
        printf "  NUM_CKPTS:           %s\n", num_ckpts
        printf "  CHECKPOINT_INTERVAL: %s\n", interval
        printf "  T:                   %s\n", T
        print ""
        
        # Calculate which checkpoint contains t=1000
        if (interval != "" && T != "") {
            target_t = 1000
            expected_ckpt = int(target_t / interval)
            printf "  Expected checkpoint for t=1000: %d (covers positions %d-%d)\n", 
                expected_ckpt, expected_ckpt * interval, (expected_ckpt + 1) * interval - 1
        }
    } else {
        print "  [No kernel config found in output]"
    }
}
' "$LOG_FILE"

echo ""
echo "=== CHECKPOINT PROCESSING LOG ==="
gawk '
/Processing checkpoint:/ {
    match($0, /Processing checkpoint: (-?[0-9]+)/, a)
    ckpt = a[1]
    in_ckpt = 1
    next
}
in_ckpt && /seg_start=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    seg_start = a[1]
    next
}
in_ckpt && /seg_end=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    seg_end = a[1]
    printf "  Checkpoint %2d: positions %4d-%4d", ckpt, seg_start, seg_end
    
    # Check if t=1000 falls in this range
    if (seg_start <= 1000 && 1000 < seg_end) {
        printf "  ← CONTAINS t=1000"
    }
    printf "\n"
    in_ckpt = 0
    next
}
END {
    if (NR == 0) {
        print "  [No checkpoint processing log found]"
    }
}
' "$LOG_FILE"

echo "=== PYTORCH REFERENCE (Ground Truth) ==="
gawk '
/=== PYTORCH REF t=1000 k=1/ { in_pt = 1; next }
in_pt && /^alpha_t_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_sum = a[1]; next }
in_pt && /^alpha_t\[0\]:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_0 = a[1]; next }
in_pt && /^beta_next_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); beta_sum = a[1]; next }
in_pt && /^beta_next\[0\]:/ { match($0, /: (-?[0-9.e+-]+)/, a); beta_0 = a[1]; next }
in_pt && /^edge_block_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); edge_sum = a[1]; next }
in_pt && /^edge_block\[0,0\]:/ { match($0, /: (-?[0-9.e+-]+)/, a); edge_00 = a[1]; next }
in_pt && /^log_joint_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_joint_sum = a[1]; next }
in_pt && /^log_joint\[0,0\]:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_joint_00 = a[1]; next }
in_pt && /^log_marginal_rel_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_marg_rel_sum = a[1]; next }
in_pt && /^marginal_unnorm_sum:/ { match($0, /: (-?[0-9.e+-]+)/, a); marg_unnorm_sum = a[1]; next }
in_pt && /local_ref/ { match($0, /: (-?[0-9.e+-]+)/, a); local_ref = a[1]; next }
in_pt && /log_norm_at_ckpt/ { match($0, /: (-?[0-9.e+-]+)/, a); log_norm = a[1]; next }
in_pt && /^log_Z:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_Z = a[1]; next }
in_pt && /^log_scale:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_scale = a[1]; next }
in_pt && /^scale:/ { match($0, /: (-?[0-9.e+-]+)/, a); scale = a[1]; next }
in_pt && /marginal_sum \(all\)/ { match($0, /: (-?[0-9.e+-]+)/, a); marg_sum = a[1]; next }
in_pt && /marginal\[0,0,0\]/ { match($0, /: (-?[0-9.e+-]+)/, a); marg_00 = a[1]; next }
in_pt && /marginal_max/ { match($0, /: (-?[0-9.e+-]+)/, a); marg_max = a[1]; next }
in_pt && /marginal_sum_over_src_total/ { match($0, /: (-?[0-9.e+-]+)/, a); sum_src = a[1]; next }
in_pt && /marginal_sum_over_src\[0\]/ { match($0, /: (-?[0-9.e+-]+)/, a); sum_src_0 = a[1]; in_pt = 0; next }
END {
    print "  Intermediate Values:"
    printf "    alpha_t_sum:                 %s\n", alpha_sum
    printf "    alpha_t[0]:                  %s\n", alpha_0
    printf "    beta_next_sum:               %s\n", beta_sum
    printf "    beta_next[0]:                %s\n", beta_0
    printf "    edge_block_sum:              %s\n", edge_sum
    printf "    edge_block[0,0]:             %s\n", edge_00
    printf "    log_joint_sum:               %s\n", log_joint_sum
    printf "    log_joint[0,0]:              %s\n", log_joint_00
    printf "    log_marginal_rel_sum:        %s\n", log_marg_rel_sum
    printf "    marginal_unnorm_sum:         %s\n", marg_unnorm_sum
    print ""
    print "  Pass 1 Statistics:"
    printf "    local_ref (global_max):      %s\n", local_ref
    printf "    log_norm_at_ckpt:            %s\n", log_norm
    printf "    log_Z:                       %s\n", log_Z
    printf "    log_scale:                   %s\n", log_scale
    printf "    scale:                       %s\n", scale
    print ""
    print "  Final Marginals:"
    printf "    marginal_sum (all):          %s\n", marg_sum
    printf "    marginal[0,0,0]:             %s\n", marg_00
    printf "    marginal_max:                %s\n", marg_max
    printf "    marginal_sum_over_src_total: %s\n", sum_src
    printf "    marginal_sum_over_src[0]:    %s\n", sum_src_0
}
' "$LOG_FILE"

echo ""
echo "=== TRITON LEVEL 1: Pass 1 Output (Global Statistics) ==="
gawk '
/=== PASS1 DEBUG t=1000 k=1 ===/ { in_pass1 = 1; next }
in_pass1 && /idx \(\) ckpt_idx=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    ckpt_idx_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) log_norm_at_ckpt=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    log_norm_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) log_Z=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    log_Z_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) global_max=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    gmax_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) global_sum_exp=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    gsum_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) log_scale=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    lscale_vals[a[1]]++
    next
}
in_pass1 && /idx \(\) scale=:/ && !/log_scale/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    scale_vals[a[1]]++
    in_pass1 = 0
    next
}
END {
    print "  ckpt_idx (deduplicated):"
    for (val in ckpt_idx_vals) printf "    %s (×%d threads)\n", val, ckpt_idx_vals[val]
    print ""
    print "  log_norm_at_ckpt (deduplicated):"
    for (val in log_norm_vals) printf "    %s (×%d threads)\n", val, log_norm_vals[val]
    print ""
    print "  log_Z (deduplicated):"
    for (val in log_Z_vals) printf "    %s (×%d threads)\n", val, log_Z_vals[val]
    print ""
    print "  global_max (deduplicated):"
    for (val in gmax_vals) printf "    %s (×%d threads)\n", val, gmax_vals[val]
    print ""
    print "  global_sum_exp (deduplicated):"
    for (val in gsum_vals) printf "    %s (×%d threads)\n", val, gsum_vals[val]
    print ""
    print "  log_scale (deduplicated):"
    for (val in lscale_vals) printf "    %s (×%d threads)\n", val, lscale_vals[val]
    print ""
    print "  scale (deduplicated):"
    for (val in scale_vals) printf "    %s (×%d threads)\n", val, scale_vals[val]
}
' "$LOG_FILE"

echo ""
echo "=== TRITON LEVEL 2: Marginal Tile (Before Sum Reduction) ==="
echo "(Aggregated across ALL tiles)"
gawk '
/=== MARGINAL TILE DEBUG t=1000 k=1 ===/ { in_tile = 1; next }
in_tile && /idx \(\) alpha_t_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_sum_vals[a[1]]++; next }
in_tile && /idx \(\) alpha_t\[0\]=:/ { match($0, /: (-?[0-9.e+-]+)/, a); alpha_0_vals[a[1]]++; next }
in_tile && /idx \(\) beta_tile_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); beta_sum_vals[a[1]]++; next }
in_tile && /idx \(\) beta_tile\[0\]=:/ { match($0, /: (-?[0-9.e+-]+)/, a); beta_0_vals[a[1]]++; next }
in_tile && /idx \(\) edge_tile_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); edge_sum_vals[a[1]]++; next }
in_tile && /idx \(\) edge_tile\[0,0\]=:/ { match($0, /: (-?[0-9.e+-]+)/, a); edge_00_vals[a[1]]++; next }
in_tile && /idx \(\) log_joint_tile_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_joint_sum_vals[a[1]]++; next }
in_tile && /idx \(\) log_joint_tile\[0,0\]=:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_joint_00_vals[a[1]]++; next }
in_tile && /idx \(\) log_marginal_rel_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); log_marg_rel_sum_vals[a[1]]++; next }
in_tile && /idx \(\) marginal_unnorm_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); marg_unnorm_sum_vals[a[1]]++; next }
in_tile && /idx \(\) marginal_tile_sum=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tile_sum_vals[a[1]]++; next }
in_tile && /idx \(\) marginal_tile\[0,0\]=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tile_00_vals[a[1]]++; next }
in_tile && /idx \(\) marginal_tile_max=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tile_max_vals[a[1]]++; next }
in_tile && /idx \(\) marginal_tile_min=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tile_min_vals[a[1]]++; in_tile = 0; next }
END {
    print "  Intermediate Values:"
    print "    alpha_t_sum (deduplicated):"
    for (val in alpha_sum_vals) printf "      %s (×%d threads)\n", val, alpha_sum_vals[val]
    print "    alpha_t[0] (deduplicated):"
    for (val in alpha_0_vals) printf "      %s (×%d threads)\n", val, alpha_0_vals[val]
    print "    beta_tile_sum (deduplicated):"
    for (val in beta_sum_vals) printf "      %s (×%d threads)\n", val, beta_sum_vals[val]
    print "    beta_tile[0] (deduplicated):"
    for (val in beta_0_vals) printf "      %s (×%d threads)\n", val, beta_0_vals[val]
    print "    edge_tile_sum (deduplicated):"
    for (val in edge_sum_vals) printf "      %s (×%d threads)\n", val, edge_sum_vals[val]
    print "    edge_tile[0,0] (deduplicated):"
    for (val in edge_00_vals) printf "      %s (×%d threads)\n", val, edge_00_vals[val]
    print "    log_joint_tile_sum (deduplicated):"
    for (val in log_joint_sum_vals) printf "      %s (×%d threads)\n", val, log_joint_sum_vals[val]
    print "    log_joint_tile[0,0] (deduplicated):"
    for (val in log_joint_00_vals) printf "      %s (×%d threads)\n", val, log_joint_00_vals[val]
    print "    log_marginal_rel_sum (deduplicated):"
    for (val in log_marg_rel_sum_vals) printf "      %s (×%d threads)\n", val, log_marg_rel_sum_vals[val]
    print "    marginal_unnorm_sum (deduplicated):"
    for (val in marg_unnorm_sum_vals) printf "      %s (×%d threads)\n", val, marg_unnorm_sum_vals[val]
    print ""
    print "  Final Marginal Tile:"
    print "    marginal_tile_sum (deduplicated):"
    for (val in tile_sum_vals) printf "      %s (×%d threads)\n", val, tile_sum_vals[val]
    print "    marginal_tile[0,0] (deduplicated):"
    for (val in tile_00_vals) printf "      %s (×%d threads)\n", val, tile_00_vals[val]
    print "    marginal_tile_max (deduplicated):"
    for (val in tile_max_vals) printf "      %s (×%d threads)\n", val, tile_max_vals[val]
    print "    marginal_tile_min (deduplicated):"
    for (val in tile_min_vals) printf "      %s (×%d threads)\n", val, tile_min_vals[val]
}
' "$LOG_FILE"

echo ""
echo "=== TRITON LEVEL 3: After Sum Reduction ==="
echo "(Aggregated across ALL tiles)"
gawk '
/=== SUM REDUCTION DEBUG t=1000 k=1 ===/ { in_sum = 1; next }
in_sum && /idx \(\) marginal_sum_src_tile_total=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    sum_total_vals[a[1]]++
    next
}
in_sum && /idx \(\) marginal_sum_src_tile\[0\]=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    sum_0_vals[a[1]]++
    in_sum = 0
    next
}
END {
    print "  marginal_sum_src_tile_total (deduplicated):"
    for (val in sum_total_vals) printf "    %s (×%d threads)\n", val, sum_total_vals[val]
    print ""
    print "  marginal_sum_src_tile[0] (deduplicated):"
    for (val in sum_0_vals) printf "    %s (×%d threads)\n", val, sum_0_vals[val]
}
' "$LOG_FILE"

echo ""
echo "=== TRITON LEVEL 4: Scatter-Sum Pattern ==="
gawk '
/=== SCATTER-SUM DEBUG t=1000 k=1 ===/ { in_scatter = 1; next }
in_scatter && /idx \(\) grad_cs_t_local_sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    grad_sum_vals[a[1]]++
    next
}
in_scatter && /idx \(\) grad_cs_t_local\[0\]=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a)
    grad_0_vals[a[1]]++
    in_scatter = 0
    next
}
END {
    print "  grad_cs_t_local_sum (deduplicated):"
    for (val in grad_sum_vals) printf "    %s (×%d threads)\n", val, grad_sum_vals[val]
    print ""
    print "  grad_cs_t_local[0] (deduplicated):"
    for (val in grad_0_vals) printf "    %s (×%d threads)\n", val, grad_0_vals[val]
}
' "$LOG_FILE"

echo ""
echo "================================================================================"
echo "CRITICAL ANALYSIS: Sum Reduction Precision Test"
echo "================================================================================"
echo ""

gawk '
BEGIN {
    # Extract values from PyTorch reference
    pt_marg_sum = 0
    pt_sum_src = 0

    # Extract values from Triton
    tr_tile_sum = 0
    tr_sum_total = 0
    tr_gmax = 0
    tr_scale = 0

    pt_gmax = 0
    pt_scale = 0
    pt_log_norm = 0
    pt_log_Z = 0
    tr_log_norm = 0
    tr_log_Z = 0
}

# PyTorch reference
/=== PYTORCH REF t=1000 k=1/ { in_pt = 1; next }
in_pt && /local_ref/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_gmax = a[1]; next }
in_pt && /log_norm_at_ckpt/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_log_norm = a[1]; next }
in_pt && /^log_Z:/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_log_Z = a[1]; next }
in_pt && /^scale:/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_scale = a[1]; next }
in_pt && /marginal_sum \(all\)/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_marg_sum = a[1]; next }
in_pt && /marginal_sum_over_src_total/ { match($0, /: (-?[0-9.e+-]+)/, a); pt_sum_src = a[1]; in_pt = 0; next }

# Triton Level 1
/=== PASS1 DEBUG t=1000 k=1 ===/ { in_tr1 = 1; next }
in_tr1 && /idx \(\) log_norm_at_ckpt=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tr_log_norm = a[1]; next }
in_tr1 && /idx \(\) log_Z=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tr_log_Z = a[1]; next }
in_tr1 && /idx \(\) global_max=:/ { match($0, /: (-?[0-9.e+-]+)/, a); tr_gmax = a[1]; next }
in_tr1 && /idx \(\) scale=:/ && !/log_scale/ { match($0, /: (-?[0-9.e+-]+)/, a); tr_scale = a[1]; in_tr1 = 0; next }

# Triton Level 2 (will show all tiles, each deduplicated)
/=== MARGINAL TILE DEBUG/ { in_tr2 = 1; next }
in_tr2 && /idx \(\) marginal_tile_sum=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    tr_tile_sums[a[1]]++;
    next
}
in_tr2 && /===/ { in_tr2 = 0; next }

# Triton Level 3 (will show all tiles, each deduplicated)
/=== SUM REDUCTION DEBUG/ { in_tr3 = 1; next }
in_tr3 && /idx \(\) marginal_sum_src_tile_total=:/ {
    match($0, /: (-?[0-9.e+-]+)/, a);
    tr_sum_totals[a[1]]++;
    next
}
in_tr3 && /===/ { in_tr3 = 0; next }

END {
    # Aggregate across all tiles
    tr_tile_sum = 0
    for (val in tr_tile_sums) {
        tr_tile_sum += val
    }
    tr_sum_total = 0
    for (val in tr_sum_totals) {
        tr_sum_total += val
    }

    print "TEST 1: Does tl.sum(marginal_tile, axis=1) preserve total?"
    print "-------------------------------------------------------------"
    if (tr_tile_sum != 0 && tr_sum_total != 0) {
        diff = tr_tile_sum - tr_sum_total
        abs_diff = (diff < 0) ? -diff : diff
        rel_error = (tr_tile_sum != 0) ? (abs_diff / tr_tile_sum * 100) : 0

        printf "  marginal_tile_sum (all tiles):          %17.10e\n", tr_tile_sum
        printf "  marginal_sum_src_tile_total (all tiles): %17.10e\n", tr_sum_total
        printf "  Difference:                  %17.10e\n", diff
        printf "  Relative error:              %7.4f%%\n", rel_error
        print ""

        if (rel_error < 0.001) {
            print "  ✓ PASS: Sum reduction is ACCURATE (error < 0.001%)"
            print "  → Issue is NOT in tl.sum(axis=1) itself"
        } else if (rel_error < 1.0) {
            print "  ⚠ WARNING: Sum reduction has SMALL error (" rel_error "%)"
            print "  → May compound over 400k operations at large scale"
        } else {
            print "  ✗ FAIL: Sum reduction has LARGE error (" rel_error "%)"
            print "  → CONFIRMED: tl.sum(axis=1) precision issue"
        }
    } else {
        print "  ⚠ SKIP: No Triton Level 2/3 data found"
    }

    print ""
    print "TEST 2: Does PyTorch sum reduction match expected total?"
    print "-------------------------------------------------------------"
    if (pt_marg_sum != 0 && pt_sum_src != 0) {
        diff_pt = pt_marg_sum - pt_sum_src
        abs_diff_pt = (diff_pt < 0) ? -diff_pt : diff_pt
        rel_error_pt = (pt_marg_sum != 0) ? (abs_diff_pt / pt_marg_sum * 100) : 0

        printf "  marginal_sum (all):          %17.10e\n", pt_marg_sum
        printf "  marginal_sum_over_src_total: %17.10e\n", pt_sum_src
        printf "  Difference:                  %17.10e\n", diff_pt
        printf "  Relative error:              %7.4f%%\n", rel_error_pt
        print ""

        if (rel_error_pt < 0.001) {
            print "  ✓ PyTorch sum reduction is accurate"
        } else {
            print "  ✗ PyTorch sum reduction has error (unexpected)"
        }
    } else {
        print "  ⚠ SKIP: No PyTorch reference data found"
    }

    print ""
    print "TEST 3: Triton vs PyTorch Comparison"
    print "-------------------------------------------------------------"
    if (tr_gmax != 0 && pt_gmax != 0) {
        log_norm_diff = tr_log_norm - pt_log_norm
        abs_log_norm_diff = (log_norm_diff < 0) ? -log_norm_diff : log_norm_diff
        rel_log_norm_error = (pt_log_norm != 0) ? (abs_log_norm_diff / pt_log_norm * 100) : 0

        log_Z_diff = tr_log_Z - pt_log_Z
        abs_log_Z_diff = (log_Z_diff < 0) ? -log_Z_diff : log_Z_diff
        rel_log_Z_error = (pt_log_Z != 0) ? (abs_log_Z_diff / pt_log_Z * 100) : 0

        gmax_diff = tr_gmax - pt_gmax
        abs_gmax_diff = (gmax_diff < 0) ? -gmax_diff : gmax_diff

        scale_diff = tr_scale - pt_scale
        abs_scale_diff = (scale_diff < 0) ? -scale_diff : scale_diff
        rel_scale_error = (pt_scale != 0) ? (abs_scale_diff / pt_scale * 100) : 0

        tile_diff = tr_tile_sum - pt_marg_sum
        abs_tile_diff = (tile_diff < 0) ? -tile_diff : tile_diff
        rel_tile_error = (pt_marg_sum != 0) ? (abs_tile_diff / pt_marg_sum * 100) : 0

        printf "  PyTorch log_norm:     %17.10e\n", pt_log_norm
        printf "  Triton log_norm:      %17.10e\n", tr_log_norm
        printf "  Difference:           %17.10e\n", log_norm_diff
        printf "  Relative error:       %7.4f%%\n", rel_log_norm_error
        print ""

        printf "  PyTorch log_Z:        %17.10e\n", pt_log_Z
        printf "  Triton log_Z:         %17.10e\n", tr_log_Z
        printf "  Difference:           %17.10e\n", log_Z_diff
        printf "  Relative error:       %7.4f%%\n", rel_log_Z_error
        print ""

        printf "  PyTorch global_max:   %17.10e\n", pt_gmax
        printf "  Triton global_max:    %17.10e\n", tr_gmax
        printf "  Difference:           %17.10e\n", gmax_diff
        print ""

        printf "  PyTorch scale:        %17.10e\n", pt_scale
        printf "  Triton scale:         %17.10e\n", tr_scale
        printf "  Difference:           %17.10e\n", scale_diff
        printf "  Relative error:       %7.4f%%\n", rel_scale_error
        print ""

        printf "  PyTorch marginal_sum: %17.10e\n", pt_marg_sum
        printf "  Triton marginal_sum:  %17.10e\n", tr_tile_sum
        printf "  Difference:           %17.10e\n", tile_diff
        printf "  Relative error:       %7.4f%%\n", rel_tile_error
        print ""

        if (abs_gmax_diff < 1e-6 && rel_scale_error < 0.1 && rel_log_norm_error < 0.01) {
            print "  ✓ Pass 1 statistics match PyTorch"
        } else {
            print "  ✗ Pass 1 diverges from PyTorch"
            print ""
            if (rel_log_norm_error > 0.01) {
                print "  → PRIMARY ISSUE: log_norm_at_ckpt mismatch (" rel_log_norm_error "%)"
                print "    - Forward pass may have incorrect log_norm_ckpts"
                print "    - OR checkpoint indexing is wrong"
                print "    - OR wrong checkpoint is being loaded"
            }
            if (abs_gmax_diff > 1e-6) {
                print "  → SECONDARY ISSUE: global_max error"
                print "    - Error is in Pass 1 online logsumexp"
            }
            if (rel_scale_error > 0.1 && rel_log_norm_error < 0.01 && abs_gmax_diff < 1e-6) {
                print "  → scale error without log_norm/global_max error"
                print "    - Check log_scale computation arithmetic"
            }
        }
        print ""

        if (rel_tile_error < 1.0) {
            print "  ✓ Marginal tile matches PyTorch (< 1% error)"
        } else {
            print "  ✗ Marginal tile diverges from PyTorch"
            print "  → Error is in marginal computation (Pass 2)"
        }
    } else {
        print "  ⚠ SKIP: Missing comparison data"
    }

    print ""
    print "TEST 4: Step-by-Step Intermediate Value Comparison"
    print "-------------------------------------------------------------"
    print "This traces the computation chain to find where divergence occurs:"
    print "  alpha + edge + beta → log_joint → exp(log_joint - global_max) → * scale → marginal"
    print ""

    # Note: Need to extract Triton intermediate values from earlier parsing
    # This section will be populated by a separate AWK pass
    print "  [Run script again with updated log to see intermediate comparisons]"

    print ""
    print "DIAGNOSIS SUMMARY"
    print "================="
    print ""

    if (tr_tile_sum != 0 && tr_sum_total != 0) {
        diff = tr_tile_sum - tr_sum_total
        abs_diff = (diff < 0) ? -diff : diff
        rel_error = (tr_tile_sum != 0) ? (abs_diff / tr_tile_sum * 100) : 0

        if (rel_error < 0.001) {
            print "→ Sum reduction (tl.sum(axis=1)) is NOT the issue"
            print "→ Error must be in:"
            print "    1. Scatter-sum pattern (Level 4)"
            print "    2. Accumulation across k iterations"
            print "    3. Pass 1 online logsumexp"
        } else {
            print "→ CONFIRMED: Sum reduction precision loss"
            print "→ Fix: Cast to float64 before sum:"
            print "    marginal_sum_src_tile = tl.sum(marginal_tile.to(tl.float64), axis=1)"
        }
    }
}
' "$LOG_FILE"

echo ""
echo "================================================================================"
echo "PYTHON SCRIPT OUTPUT SUMMARY"
echo "================================================================================"
echo ""

echo "=== PYTORCH REFERENCE COMPARISON ==="
grep -A 10 "Comparison vs PyTorch Reference" "$LOG_FILE" 2>/dev/null | head -12
echo ""

echo "=== FULL TENSOR COMPARISON VS PYTORCH ==="
grep -A 6 "Full Tensor Comparison vs PyTorch" "$LOG_FILE" 2>/dev/null | head -8
echo ""

echo "=== RUN-TO-RUN DIFFERENCES ==="
grep -A 30 "Difference Analysis (Run-to-Run)" "$LOG_FILE" 2>/dev/null | head -35
echo ""

echo "=== FINAL VERDICT ==="
grep -E "\[PASS\]|\[FAIL\]" "$LOG_FILE" 2>/dev/null | tail -5
echo ""

echo "================================================================================"
echo "DIAGNOSIS"
echo "================================================================================"
echo ""

# Check for non-determinism patterns
if grep -q "\[FAIL\] Non-determinism detected" "$LOG_FILE" 2>/dev/null; then
    echo "[FAIL] Non-determinism detected in backward pass"
    echo ""
    echo "Check the sections above:"
    echo "  1. If kernel debug values show multiple unique values -> Triton kernel issue"
    echo "  2. If kernel debug values are consistent but Python shows run-to-run diff -> atomic_add race"
    echo "  3. If Triton matches PyTorch but runs differ -> Forward pass providing different checkpoints"
elif grep -q "\[PASS\] All.*runs produced identical results" "$LOG_FILE" 2>/dev/null; then
    echo "[PASS] Backward pass is deterministic"
    echo ""
    # Check if there's still error vs PyTorch
    if grep -q "max_diff_db=.*e+00" "$LOG_FILE" 2>/dev/null; then
        echo "WARNING: Deterministic but may have correctness issues vs PyTorch reference"
        echo "         Check 'Full Tensor Comparison vs PyTorch' section above"
    fi
else
    echo "[UNKNOWN] Could not determine verdict from log file"
fi
echo ""
