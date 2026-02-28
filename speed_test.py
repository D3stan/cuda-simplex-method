#!/usr/bin/env python3
"""
Speed comparison: HiGHS (CPU) vs CUDA Simplex solver.

Runs every MPS file in tests/ through both solvers multiple times,
measures wall-clock time, and prints a summary table with speedup ratios.
"""

import glob
import os
import re
import statistics
import subprocess
import sys
import time

try:
    import highspy
except ImportError:
    print("ERROR: highspy not installed. Install with: pip install highspy")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTS_DIR = os.path.join(SCRIPT_DIR, "tests")
CUDA_BINARY = os.path.join(SCRIPT_DIR, "simplex.out")

WARMUP_RUNS = 2       # discarded warm-up iterations
TIMED_RUNS = 10       # measured iterations per problem
TOLERANCE = 1e-4      # for correctness check


# ═══════════════════════════════════════════════════════════════════════════════
# HiGHS solver
# ═══════════════════════════════════════════════════════════════════════════════
def solve_highs(mps_path: str):
    """Solve with HiGHS; return (objective, status_str, elapsed_sec)."""
    h = highspy.Highs()
    h.silent()
    st = h.readModel(mps_path)
    if st != highspy.HighsStatus.kOk:
        return None, "READ_ERROR", 0.0

    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0

    model_status = h.getModelStatus()
    obj = h.getInfoValue("objective_function_value")[1]

    # Map HiGHS status to canonical strings matching CUDA solver output
    status_str = str(model_status)
    if "kOptimal" in status_str:
        status = "OPTIMAL"
    elif "kUnbounded" in status_str:
        status = "UNBOUNDED"
    elif "kInfeasible" in status_str:
        status = "INFEASIBLE"
    else:
        status = status_str
    return obj, status, elapsed


def bench_highs(mps_path: str, warmup: int, runs: int):
    """Benchmark HiGHS: warm up, then collect `runs` timing samples."""
    for _ in range(warmup):
        solve_highs(mps_path)

    times = []
    obj = None
    status = None
    for _ in range(runs):
        obj, status, t = solve_highs(mps_path)
        times.append(t)
    return obj, status, times


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA Simplex solver
# ═══════════════════════════════════════════════════════════════════════════════
def solve_cuda(binary: str, mps_path: str):
    """Run CUDA simplex; return (objective, status_str, elapsed_sec).

    Elapsed time is read from the program's own internal timer
    (hpc_gettime), which measures only the computation and excludes
    file I/O and output.  Falls back to external wall-clock timing
    if the internal timer line is not found.
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [binary, "-s", mps_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    external_elapsed = time.perf_counter() - t0

    output = result.stdout

    # Parse status from stdout (printed even on non-zero exit code)
    status = "UNKNOWN"
    if "Status: OPTIMAL" in output:
        status = "OPTIMAL"
    elif "Status: INFEASIBLE" in output or "INFEASIBLE" in output:
        status = "INFEASIBLE"
    elif "Status: UNBOUNDED" in output or "UNBOUNDED" in output:
        status = "UNBOUNDED"
    elif result.returncode != 0:
        status = "ERROR"

    m = re.search(r"Objective Value:\s*([-\d.eE+]+)", output)
    obj = float(m.group(1)) if m else None

    # Prefer internal elapsed time reported by hpc_gettime() inside the solver
    m_time = re.search(r"Elapsed time:\s*([\d.eE+-]+)\s*seconds", output)
    elapsed = float(m_time.group(1)) if m_time else external_elapsed

    return obj, status, elapsed


def bench_cuda(binary: str, mps_path: str, warmup: int, runs: int):
    """Benchmark CUDA simplex: warm up, then collect `runs` timing samples."""
    for _ in range(warmup):
        solve_cuda(binary, mps_path)

    times = []
    obj = None
    status = None
    for _ in range(runs):
        obj, status, t = solve_cuda(binary, mps_path)
        times.append(t)
    return obj, status, times


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_ms(sec: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{sec * 1000:.3f}"


def stats_summary(times: list) -> dict:
    """Return mean, median, min, max, stdev of a list of seconds."""
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.isfile(CUDA_BINARY):
        print(f"ERROR: CUDA binary not found at {CUDA_BINARY}")
        sys.exit(1)

    # Accept directories or individual MPS files as CLI arguments
    # Default: tests/
    targets = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_TESTS_DIR]
    mps_files = []
    for t in targets:
        if os.path.isdir(t):
            mps_files.extend(glob.glob(os.path.join(t, "*.mps")))
        elif os.path.isfile(t) and t.endswith(".mps"):
            mps_files.append(t)
        else:
            print(f"WARNING: skipping {t} (not a dir or .mps file)")
    mps_files = sorted(mps_files)
    if not mps_files:
        print(f"No MPS files found in: {targets}")
        sys.exit(1)

    print("=" * 90)
    print("  SPEED TEST: HiGHS (CPU)  vs  CUDA Simplex (GPU)")
    print(f"  Warm-up runs: {WARMUP_RUNS}  |  Timed runs: {TIMED_RUNS}")
    print("=" * 90)

    # Header
    hdr = (
        f"{'Problem':<20} "
        f"{'HiGHS (ms)':>12} {'CUDA (ms)':>12} {'Speedup':>10} "
        f"{'Match':>7} {'Status':>10}"
    )
    print(f"\n{hdr}")
    print("-" * 90)

    results = []

    for mps_path in mps_files:
        name = os.path.splitext(os.path.basename(mps_path))[0]
        sys.stdout.write(f"  {name:<18} ")
        sys.stdout.flush()

        # ── Benchmark HiGHS ──
        h_obj, h_status, h_times = bench_highs(mps_path, WARMUP_RUNS, TIMED_RUNS)
        h_stats = stats_summary(h_times)

        # ── Benchmark CUDA ──
        c_obj, c_status, c_times = bench_cuda(CUDA_BINARY, mps_path, WARMUP_RUNS, TIMED_RUNS)
        c_stats = stats_summary(c_times)

        # ── Correctness check ──
        if h_obj is not None and c_obj is not None:
            match = abs(h_obj - c_obj) < TOLERANCE and h_status == c_status
        else:
            match = h_status == c_status

        # ── Speedup (HiGHS time / CUDA time; >1 means CUDA faster) ──
        if c_stats["mean"] > 0:
            speedup = h_stats["mean"] / c_stats["mean"]
        else:
            speedup = float("inf")

        row = {
            "name": name,
            "h_mean": h_stats["mean"],
            "c_mean": c_stats["mean"],
            "speedup": speedup,
            "match": match,
            "status": h_status,
            "h_stats": h_stats,
            "c_stats": c_stats,
            "h_obj": h_obj,
            "c_obj": c_obj,
        }
        results.append(row)

        # Print compact row
        sp_str = f"{speedup:.2f}x" if speedup != float("inf") else "inf"
        winner = "CUDA" if speedup > 1.0 else "HiGHS"
        print(
            f"{fmt_ms(h_stats['mean']):>12} "
            f"{fmt_ms(c_stats['mean']):>12} "
            f"{sp_str:>10} "
            f"{'OK' if match else 'FAIL':>7} "
            f"{h_status:>10}"
        )

    # ── Detailed stats ──
    print("\n" + "=" * 90)
    print("  DETAILED STATISTICS (milliseconds)")
    print("=" * 90)

    detail_hdr = (
        f"{'Problem':<20} {'Solver':<8} "
        f"{'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}"
    )
    print(f"\n{detail_hdr}")
    print("-" * 90)

    for r in results:
        hs, cs = r["h_stats"], r["c_stats"]
        print(
            f"{r['name']:<20} {'HiGHS':<8} "
            f"{fmt_ms(hs['mean']):>10} {fmt_ms(hs['median']):>10} "
            f"{fmt_ms(hs['min']):>10} {fmt_ms(hs['max']):>10} {fmt_ms(hs['stdev']):>10}"
        )
        print(
            f"{'':<20} {'CUDA':<8} "
            f"{fmt_ms(cs['mean']):>10} {fmt_ms(cs['median']):>10} "
            f"{fmt_ms(cs['min']):>10} {fmt_ms(cs['max']):>10} {fmt_ms(cs['stdev']):>10}"
        )

    # ── Objective values ──
    print("\n" + "=" * 90)
    print("  OBJECTIVE VALUES")
    print("=" * 90)
    print(f"\n{'Problem':<20} {'HiGHS Obj':>16} {'CUDA Obj':>16} {'Diff':>12} {'Match':>7}")
    print("-" * 75)
    for r in results:
        h_o = r["h_obj"]
        c_o = r["c_obj"]
        h_str = f"{h_o:.6f}" if h_o is not None else "N/A"
        c_str = f"{c_o:.6f}" if c_o is not None else "N/A"
        if h_o is not None and c_o is not None:
            diff = abs(h_o - c_o)
            d_str = f"{diff:.2e}"
        else:
            d_str = "N/A"
        print(f"{r['name']:<20} {h_str:>16} {c_str:>16} {d_str:>12} {'OK' if r['match'] else 'FAIL':>7}")

    # ── Summary ──
    total_highs = sum(r["h_mean"] for r in results)
    total_cuda = sum(r["c_mean"] for r in results)
    total_speedup = total_highs / total_cuda if total_cuda > 0 else float("inf")
    all_match = all(r["match"] for r in results)

    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  Problems tested:      {len(results)}")
    print(f"  All results match:    {'YES' if all_match else 'NO'}")
    print(f"  Total HiGHS time:     {fmt_ms(total_highs)} ms")
    print(f"  Total CUDA time:      {fmt_ms(total_cuda)} ms")
    sp_str = f"{total_speedup:.2f}x" if total_speedup != float("inf") else "inf"
    faster = "CUDA" if total_speedup > 1.0 else "HiGHS"
    print(f"  Overall speedup:      {sp_str} ({faster} faster)")
    print("=" * 90)

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
