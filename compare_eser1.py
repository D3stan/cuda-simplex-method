#!/usr/bin/env python3
"""
Compare HiGHS solver results with CUDA Simplex solver for Exercise 1.

Uses HiGHS Python API (highspy) to solve the MPS file and then runs
the CUDA simplex solver, comparing objective values and variable solutions.

Reference: https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/
"""

import subprocess
import re
import sys
import os

try:
    import highspy
except ImportError:
    print("ERROR: highspy not installed. Install with: pip install highspy")
    sys.exit(1)


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPS_FILE = os.path.join(SCRIPT_DIR, "tests", "eser1.mps")
CUDA_BINARY = os.path.join(SCRIPT_DIR, "simplex.out")

TOLERANCE = 1e-4  # tolerance for comparing floating-point results


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Solve with HiGHS
# ═══════════════════════════════════════════════════════════════════════════════
def solve_with_highs(mps_path: str) -> dict:
    """Solve the LP from an MPS file using HiGHS and return results."""
    h = highspy.Highs()
    h.silent()  # suppress HiGHS output

    status = h.readModel(mps_path)
    assert status == highspy.HighsStatus.kOk, f"Failed to read MPS file: {mps_path}"

    h.run()

    model_status = h.getModelStatus()
    info = h.getInfoValue("objective_function_value")
    obj_value = info[1]

    num_cols = h.getNumCol()
    solution = h.getSolution()
    col_values = list(solution.col_value)

    # Get variable names
    var_names = []
    for i in range(num_cols):
        name = h.getColName(i)
        var_names.append(name[1] if isinstance(name, tuple) else name)

    is_optimal = (
        model_status == highspy.HighsModelStatus.kOptimal
        or str(model_status) == "HighsModelStatus.kOptimal"
    )

    return {
        "status": "OPTIMAL" if is_optimal else f"{model_status}",
        "objective": obj_value,
        "variables": dict(zip(var_names, col_values)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Solve with CUDA Simplex
# ═══════════════════════════════════════════════════════════════════════════════
def solve_with_cuda(binary: str, mps_path: str) -> dict:
    """Run the CUDA simplex binary and parse its output."""
    if not os.path.isfile(binary):
        return {"error": f"CUDA binary not found: {binary}"}

    result = subprocess.run(
        [binary, mps_path],
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout
    if result.returncode != 0:
        return {"error": f"CUDA solver exited with code {result.returncode}\n{output}"}

    # Parse status
    status = "UNKNOWN"
    if "Status: OPTIMAL" in output:
        status = "OPTIMAL"
    elif "INFEASIBLE" in output:
        status = "INFEASIBLE"
    elif "UNBOUNDED" in output:
        status = "UNBOUNDED"

    # Parse objective value
    obj_match = re.search(r"Objective Value:\s*([-\d.]+)", output)
    obj_value = float(obj_match.group(1)) if obj_match else None

    # Parse variable values
    variables = {}
    for m in re.finditer(r"^\s*(\S+)\s*=\s*([-\d.]+)", output, re.MULTILINE):
        variables[m.group(1)] = float(m.group(2))

    return {
        "status": status,
        "objective": obj_value,
        "variables": variables,
        "raw_output": output,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Compare results
# ═══════════════════════════════════════════════════════════════════════════════
def compare(highs_res: dict, cuda_res: dict, tol: float = TOLERANCE):
    """Print a side-by-side comparison and flag mismatches."""
    print("=" * 65)
    print(" COMPARISON: HiGHS  vs  CUDA Simplex  (eser1.mps)")
    print("=" * 65)

    # --- Status ---
    h_status = highs_res["status"]
    c_status = cuda_res.get("status", cuda_res.get("error", "ERROR"))
    match_status = h_status == c_status
    print(f"\n{'Status':<25} {'HiGHS':<20} {'CUDA':<20} {'':>5}")
    print(f"{'':.<25} {h_status:<20} {c_status:<20} {'OK' if match_status else 'MISMATCH':>5}")

    # --- Objective ---
    h_obj = highs_res["objective"]
    c_obj = cuda_res.get("objective")
    if c_obj is not None:
        obj_diff = abs(h_obj - c_obj)
        match_obj = obj_diff < tol
    else:
        obj_diff = float("inf")
        match_obj = False

    print(f"\n{'Objective Value':<25} {'HiGHS':<20} {'CUDA':<20} {'Diff':>10}")
    print(
        f"{'':.<25} {h_obj:<20.6f} "
        f"{c_obj if c_obj is not None else 'N/A':<20} "
        f"{obj_diff:>10.2e}  {'OK' if match_obj else 'MISMATCH'}"
    )

    # --- Variables ---
    # Use HiGHS variable names as the canonical set (CUDA may omit zero-valued vars)
    all_vars = sorted(highs_res["variables"].keys())
    print(f"\n{'Variable':<25} {'HiGHS':<20} {'CUDA':<20} {'Diff':>10}")
    print("-" * 65)

    all_match = match_status and match_obj
    for var in all_vars:
        h_val = highs_res["variables"].get(var)
        c_val = cuda_res.get("variables", {}).get(var)

        h_str = f"{h_val:.6f}" if h_val is not None else "N/A"
        c_str = f"{c_val:.6f}" if c_val is not None else "N/A"

        # CUDA solver may omit zero-valued variables from output
        if c_val is None:
            c_val = 0.0
            c_str = f"{c_val:.6f} (implicit)"

        if h_val is not None and c_val is not None:
            diff = abs(h_val - c_val)
            ok = diff < tol
        else:
            diff = float("inf")
            ok = False

        all_match = all_match and ok
        print(f"{var:<25} {h_str:<20} {c_str:<20} {diff:>10.2e}  {'OK' if ok else 'MISMATCH'}")

    # --- Expected solution (from MPS comment) ---
    print(f"\n{'─' * 65}")
    print("Expected (from MPS comments): cost = -7.0000, X1 = 3.0, X2 = 1.0")
    print(f"{'─' * 65}")

    # --- Verdict ---
    print()
    if all_match:
        print(">>> PASS: HiGHS and CUDA Simplex solutions MATCH within tolerance.")
    else:
        print(">>> FAIL: Solutions DIFFER beyond tolerance.")

    return all_match


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Solving eser1.mps with HiGHS …")
    highs_result = solve_with_highs(MPS_FILE)

    print("Solving eser1.mps with CUDA Simplex …")
    cuda_result = solve_with_cuda(CUDA_BINARY, MPS_FILE)

    if "error" in cuda_result:
        print(f"\nCUDA Solver Error: {cuda_result['error']}")
        print("\nHiGHS result only:")
        print(f"  Status:    {highs_result['status']}")
        print(f"  Objective: {highs_result['objective']:.6f}")
        for var, val in highs_result["variables"].items():
            print(f"  {var} = {val:.6f}")
        sys.exit(1)

    ok = compare(highs_result, cuda_result)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
