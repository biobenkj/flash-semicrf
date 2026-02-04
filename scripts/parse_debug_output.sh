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
