#!/usr/bin/env python3
"""Benchmark HiGHS (CPU) vs CUDA simplex (GPU) on Netlib .mps datasets.

Requires: pip install highspy
"""

import argparse
import csv
import glob
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

STATUS_RE = re.compile(r"Status:\s*(OPTIMAL|INFEASIBLE|UNBOUNDED|TIMEOUT|ERROR)")
OBJ_RE = re.compile(r"Objective Value:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
ELAPSED_RE = re.compile(r"Elapsed time:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*seconds")


def cmd_output(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return "N/A"


def machine_info() -> Dict[str, str]:
    return {
        "kernel": cmd_output(["uname", "-a"]),
        "os_release": cmd_output(["bash", "-lc", "cat /etc/os-release"]),
        "cpu": cmd_output(["bash", "-lc", "lscpu | sed -n '1,20p'"]),
        "gpu": cmd_output([
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader",
        ]),
        "nvcc": cmd_output(["bash", "-lc", "nvcc --version | tail -n 4"]),
        "python": platform.python_version(),
    }


def run_cuda(binary: str, problem_path: str, timeout_sec: int) -> Tuple[str, Optional[float], Optional[float], int]:
    try:
        proc = subprocess.run(
            [binary, "-s", problem_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, None, 124
    out = proc.stdout

    status_match = STATUS_RE.search(out)
    status = status_match.group(1) if status_match else ("ERROR" if proc.returncode != 0 else "UNKNOWN")

    obj_match = OBJ_RE.search(out)
    obj = float(obj_match.group(1)) if obj_match else None

    elapsed_match = ELAPSED_RE.search(out)
    elapsed = float(elapsed_match.group(1)) if elapsed_match else None

    return status, obj, elapsed, proc.returncode


def run_highs(mps_path: str) -> Tuple[str, Optional[float], float]:
    import highspy  # type: ignore

    h = highspy.Highs()
    h.silent()

    st = h.readModel(mps_path)
    if st != highspy.HighsStatus.kOk:
        return "READ_ERROR", None, 0.0

    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0

    status_repr = str(h.getModelStatus())
    if "kOptimal" in status_repr:
        status = "OPTIMAL"
    elif "kUnbounded" in status_repr:
        status = "UNBOUNDED"
    elif "kInfeasible" in status_repr:
        status = "INFEASIBLE"
    else:
        status = status_repr

    obj = h.getInfoValue("objective_function_value")[1]
    return status, obj, elapsed


def summarize(times: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def fmt_ms(sec: float) -> str:
    return f"{sec * 1000:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="HiGHS vs CUDA benchmark on Netlib .mps datasets")
    parser.add_argument("--binary", default="simplex.out", help="Path to CUDA solver binary")
    parser.add_argument("--netlib-dir", default="data/netlib", help="Directory with .mps files")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs per solver per dataset")
    parser.add_argument("--runs", type=int, default=2, help="Timed runs per solver per dataset")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout (seconds) for one CUDA run")
    parser.add_argument("--json-out", default="", help="Optional output JSON file")
    parser.add_argument("--csv-out", default="", help="Optional output CSV file")
    args = parser.parse_args()

    try:
        import highspy  # type: ignore

        _ = highspy
    except Exception:
        print("ERROR: highspy is not installed. Install with: pip install highspy")
        return 2

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    binary = args.binary if os.path.isabs(args.binary) else os.path.join(repo_root, args.binary)
    netlib_dir = args.netlib_dir if os.path.isabs(args.netlib_dir) else os.path.join(repo_root, args.netlib_dir)

    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}")
        return 1

    mps_files = sorted(glob.glob(os.path.join(netlib_dir, "*.mps")))
    if not mps_files:
        print(f"ERROR: no .mps files found in {netlib_dir}")
        return 1

    info = machine_info()

    print("=" * 124)
    print("Netlib Benchmark: HiGHS (CPU) vs CUDA simplex (GPU)")
    print(f"Binary: {binary}")
    print(f"Datasets: {len(mps_files)} from {netlib_dir}")
    print(f"Warm-up runs: {args.warmup} | Timed runs: {args.runs}")
    print("=" * 124)
    print(f"{'Dataset':<14} {'CPU(ms)':>10} {'GPU(ms)':>10} {'Speedup':>9} {'CPU Obj':>16} {'GPU Obj':>16} {'AbsDiff':>12} {'RelDiff':>12} {'Match':>8}")
    print("-" * 124)

    rows = []

    for mps_path in mps_files:
        name = os.path.basename(mps_path)

        for _ in range(args.warmup):
            run_highs(mps_path)
            run_cuda(binary, mps_path, args.timeout)

        highs_times: List[float] = []
        cuda_times: List[float] = []
        h_status = "UNKNOWN"
        c_status = "UNKNOWN"
        h_obj: Optional[float] = None
        c_obj: Optional[float] = None

        for _ in range(args.runs):
            h_status, h_obj, h_elapsed = run_highs(mps_path)
            highs_times.append(h_elapsed)

            c_status, c_obj, c_elapsed, _ = run_cuda(binary, mps_path, args.timeout)
            if c_elapsed is not None:
                cuda_times.append(c_elapsed)

        if not cuda_times:
            print(f"{name:<14} {'N/A':>10} {'N/A':>10} {'N/A':>9} {'N/A':>16} {'N/A':>16} {'N/A':>12} {'N/A':>12} {'NO':>8}")
            rows.append({"dataset": name, "error": "cuda_no_timing"})
            continue

        hs = summarize(highs_times)
        cs = summarize(cuda_times)

        speedup = hs["mean"] / cs["mean"] if cs["mean"] > 0 else float("inf")
        abs_diff = abs(h_obj - c_obj) if (h_obj is not None and c_obj is not None) else None
        rel_diff = (abs_diff / max(1.0, abs(h_obj))) if (abs_diff is not None and h_obj is not None) else None

        match = h_status == c_status

        h_obj_str = f"{h_obj:.6f}" if h_obj is not None else "N/A"
        c_obj_str = f"{c_obj:.6f}" if c_obj is not None else "N/A"
        abs_str = f"{abs_diff:.3e}" if abs_diff is not None else "N/A"
        rel_str = f"{rel_diff:.3e}" if rel_diff is not None else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup != float("inf") else "inf"

        print(
            f"{name:<14} {fmt_ms(hs['mean']):>10} {fmt_ms(cs['mean']):>10} {speedup_str:>9} "
            f"{h_obj_str:>16} {c_obj_str:>16} {abs_str:>12} {rel_str:>12} {('YES' if match else 'NO'):>8}"
        )

        rows.append(
            {
                "dataset": name,
                "highs": {
                    "status": h_status,
                    "objective": h_obj,
                    "timing": {
                        "mean_s": hs["mean"],
                        "median_s": hs["median"],
                        "min_s": hs["min"],
                        "max_s": hs["max"],
                        "stdev_s": hs["stdev"],
                    },
                },
                "cuda": {
                    "status": c_status,
                    "objective": c_obj,
                    "timing": {
                        "mean_s": cs["mean"],
                        "median_s": cs["median"],
                        "min_s": cs["min"],
                        "max_s": cs["max"],
                        "stdev_s": cs["stdev"],
                    },
                },
                "speedup_highs_over_cuda": speedup,
                "objective_abs_diff": abs_diff,
                "objective_rel_diff": rel_diff,
                "status_match": match,
            }
        )

    valid_rows = [r for r in rows if "highs" in r and "cuda" in r]
    total_h = sum(r["highs"]["timing"]["mean_s"] for r in valid_rows)
    total_c = sum(r["cuda"]["timing"]["mean_s"] for r in valid_rows)
    total_speedup = total_h / total_c if total_c > 0 else None

    rel_errors = [r["objective_rel_diff"] for r in valid_rows if r["objective_rel_diff"] is not None]
    max_rel_error = max(rel_errors) if rel_errors else None
    mean_rel_error = statistics.mean(rel_errors) if rel_errors else None

    print("-" * 124)
    print(f"Compared datasets: {len(valid_rows)}/{len(rows)}")
    print(f"Total CPU mean time: {total_h * 1000:.3f} ms")
    print(f"Total GPU mean time: {total_c * 1000:.3f} ms")
    print(f"Total speedup (CPU/GPU): {(f'{total_speedup:.2f}x' if total_speedup is not None else 'N/A')}")
    print(f"Max relative objective error: {(f'{max_rel_error:.3e}' if max_rel_error is not None else 'N/A')}")
    print(f"Mean relative objective error: {(f'{mean_rel_error:.3e}' if mean_rel_error is not None else 'N/A')}")
    print("=" * 124)

    output_obj = {
        "benchmark": "netlib_highs_vs_cuda",
        "binary": binary,
        "netlib_dir": netlib_dir,
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "machine_info": info,
        "results": rows,
        "summary": {
            "dataset_count": len(rows),
            "compared_count": len(valid_rows),
            "total_cpu_mean_s": total_h,
            "total_gpu_mean_s": total_c,
            "total_speedup_cpu_over_gpu": total_speedup,
            "max_rel_error": max_rel_error,
            "mean_rel_error": mean_rel_error,
        },
    }

    if args.json_out:
        json_path = args.json_out if os.path.isabs(args.json_out) else os.path.join(repo_root, args.json_out)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, indent=2)
        print(f"JSON written to: {json_path}")

    if args.csv_out:
        csv_path = args.csv_out if os.path.isabs(args.csv_out) else os.path.join(repo_root, args.csv_out)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "cpu_status",
                "gpu_status",
                "cpu_objective",
                "gpu_objective",
                "cpu_mean_s",
                "gpu_mean_s",
                "speedup_cpu_over_gpu",
                "objective_abs_diff",
                "objective_rel_diff",
                "status_match",
            ])
            for r in valid_rows:
                writer.writerow([
                    r["dataset"],
                    r["highs"]["status"],
                    r["cuda"]["status"],
                    r["highs"]["objective"],
                    r["cuda"]["objective"],
                    r["highs"]["timing"]["mean_s"],
                    r["cuda"]["timing"]["mean_s"],
                    r["speedup_highs_over_cuda"],
                    r["objective_abs_diff"],
                    r["objective_rel_diff"],
                    r["status_match"],
                ])
        print(f"CSV written to: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
