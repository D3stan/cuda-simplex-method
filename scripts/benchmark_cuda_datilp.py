#!/usr/bin/env python3
"""Benchmark CUDA-only on Dati-LP .dat datasets.

Runs simplex.out multiple times per dataset and reports timing statistics.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

STATUS_RE = re.compile(r"Status:\s*(OPTIMAL|INFEASIBLE|UNBOUNDED|TIMEOUT|ERROR)")
OBJ_RE = re.compile(r"Objective Value:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
ELAPSED_RE = re.compile(r"Elapsed time:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*seconds")


def run_cuda(binary: str, problem_path: str, timeout_sec: int) -> Tuple[str, Optional[float], Optional[float], int]:
    proc = subprocess.run(
        [binary, "-s", problem_path],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    out = proc.stdout

    status_match = STATUS_RE.search(out)
    status = status_match.group(1) if status_match else ("ERROR" if proc.returncode != 0 else "UNKNOWN")

    obj_match = OBJ_RE.search(out)
    obj = float(obj_match.group(1)) if obj_match else None

    elapsed_match = ELAPSED_RE.search(out)
    elapsed = float(elapsed_match.group(1)) if elapsed_match else None

    return status, obj, elapsed, proc.returncode


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
    parser = argparse.ArgumentParser(description="CUDA-only benchmark on Dati-LP datasets")
    parser.add_argument("--binary", default="simplex.out", help="Path to CUDA solver binary")
    parser.add_argument("--data-dir", default="data/Dati-LP", help="Directory with .dat files")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs per dataset")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per dataset")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout (seconds) for one solver run")
    parser.add_argument("--json-out", default="", help="Optional output JSON file")
    parser.add_argument("--csv-out", default="", help="Optional output CSV file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    binary = args.binary if os.path.isabs(args.binary) else os.path.join(repo_root, args.binary)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(repo_root, args.data_dir)

    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}")
        return 1

    dat_files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
    if not dat_files:
        print(f"ERROR: no .dat files found in {data_dir}")
        return 1

    print("=" * 96)
    print("CUDA Benchmark on Dati-LP")
    print(f"Binary: {binary}")
    print(f"Datasets: {len(dat_files)} from {data_dir}")
    print(f"Warm-up runs: {args.warmup} | Timed runs: {args.runs}")
    print("=" * 96)
    print(f"{'Dataset':<18} {'Status':<12} {'ObjValue':>14} {'Mean(ms)':>10} {'Med(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Std(ms)':>10}")
    print("-" * 96)

    rows = []
    ok_count = 0

    for dat_path in dat_files:
        name = os.path.basename(dat_path)

        for _ in range(args.warmup):
            run_cuda(binary, dat_path, args.timeout)

        times: List[float] = []
        status = "UNKNOWN"
        obj: Optional[float] = None

        for _ in range(args.runs):
            status, obj, elapsed, _ = run_cuda(binary, dat_path, args.timeout)
            if elapsed is not None:
                times.append(elapsed)

        if not times:
            print(f"{name:<18} {status:<12} {'N/A':>14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            rows.append({"dataset": name, "status": status, "objective": obj, "timing": None})
            continue

        stats = summarize(times)
        obj_str = f"{obj:.6f}" if obj is not None else "N/A"
        print(
            f"{name:<18} {status:<12} {obj_str:>14} {fmt_ms(stats['mean']):>10} {fmt_ms(stats['median']):>10} "
            f"{fmt_ms(stats['min']):>10} {fmt_ms(stats['max']):>10} {fmt_ms(stats['stdev']):>10}"
        )

        if status == "OPTIMAL":
            ok_count += 1

        rows.append(
            {
                "dataset": name,
                "status": status,
                "objective": obj,
                "timing": {
                    "mean_s": stats["mean"],
                    "median_s": stats["median"],
                    "min_s": stats["min"],
                    "max_s": stats["max"],
                    "stdev_s": stats["stdev"],
                },
            }
        )

    mean_total_s = sum(r["timing"]["mean_s"] for r in rows if r["timing"] is not None)

    print("-" * 96)
    print(f"Datasets solved with OPTIMAL status: {ok_count}/{len(rows)}")
    print(f"Total mean time across datasets: {mean_total_s * 1000:.3f} ms")
    print("=" * 96)

    output_obj = {
        "benchmark": "cuda_only_datilp",
        "binary": binary,
        "data_dir": data_dir,
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "datasets": rows,
        "summary": {
            "dataset_count": len(rows),
            "optimal_count": ok_count,
            "total_mean_time_s": mean_total_s,
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
                "status",
                "objective",
                "mean_s",
                "median_s",
                "min_s",
                "max_s",
                "stdev_s",
            ])
            for r in rows:
                timing = r["timing"] or {}
                writer.writerow([
                    r["dataset"],
                    r["status"],
                    r["objective"],
                    timing.get("mean_s"),
                    timing.get("median_s"),
                    timing.get("min_s"),
                    timing.get("max_s"),
                    timing.get("stdev_s"),
                ])
        print(f"CSV written to: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
