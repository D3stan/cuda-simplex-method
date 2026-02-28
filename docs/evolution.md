# Evolution Report: simplex.cu

**7 commits** — from 1,202 lines to 1,990 lines (+745 insertions, −232 deletions). All work happened on **2026-02-28**.

---

## Commit 1 — `00d449e` — *First implementation*

The baseline: a complete, working **CUDA Two-Phase Simplex solver** in a single file (~1,202 lines). It already included:

- **MPS parser** — basic `sscanf`-based parser handling `ROWS`, `COLUMNS`, `RHS`, `OBJSENSE`, `ENDATA` sections. Used a two-pass approach (re-opened the file to extract objective coefficients).
- **CUDA kernels** for the core simplex operations:
  - `kernelFindPivotColumn` — parallel reduction to find the most negative reduced cost.
  - `kernelFindPivotRow` — parallel minimum-ratio test.
  - `kernelPivot` — full tableau pivot operation.
  - `kernelCachePivotData` — caches pivot row/column for the pivot step.
- **Two-Phase method** — Phase 1 with artificial variables to find a BFS, Phase 2 to optimize.
- **Tableau struct** with both host and device memory, basic variable tracking.
- A hardcoded test problem (max 3x₁ + 2x₂) as fallback.

**Limitations at this point:** No bounds handling, no RANGES, no integer markers, fragile whitespace-based parsing, no verbosity control, only 1,000 max iterations.

---

## Commit 2 — `e7f573b` — *First fixes for simplex impl*

**Massive overhaul of the MPS parser and addition of bound/range preprocessing.** This is by far the largest commit (~700 lines changed). Key improvements:

1. **Robust fixed-column MPS parsing** — replaced all `sscanf`-based parsing with proper fixed-column field extraction (`extractMPSField`), respecting the MPS standard column layout (cols 1–2, 4–11, 14–21, 24–35, 39–46, 49–60).
2. **Single-pass parsing** — eliminated the second file read; objective coefficients are now captured during the first pass via `objCoeffsTemp`.
3. **Dynamic memory allocation** — all row, variable, and coefficient arrays now grow dynamically (doubling capacity) instead of using fixed `maxVars = 1000` limits.
4. **BOUNDS section support** — full handling of `LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`, `LI`, `UI` bound types.
5. **RANGES section support** — parses range values and stores them per constraint.
6. **Integer MARKER handling** — detects `'INTORG'`/`'INTEND'` markers in the COLUMNS section to flag integer variables.
7. **Objective constant** — captures RHS entries for the N (objective) row as `objConstant`, factors it into the final objective value.
8. **`preprocessBounds()` function** — added a complete preprocessing pipeline:
   - Shifts variables with non-zero finite lower bounds (x'_j = x_j - l_j).
   - Splits free variables (l = -∞) into x = x⁺ - x⁻.
   - Adds explicit upper-bound constraints (x_j ≤ u_j).
   - Expands range constraints into pairs of inequalities.
9. **New `LPProblem` fields** — `lowerBounds`, `upperBounds`, `objConstant`, `isInteger`, `rangeValues`.
10. **Improved `freeLPProblem`** — properly frees all new fields.

---

## Commit 3 — `a90a841` — *Fixed bug with exercise 9*

**Critical correctness fix for negative RHS handling.** The constraint-flipping logic was moved to a **pre-pass** (before counting slack/surplus/artificial variables):

- **Before:** RHS sign flipping happened *during* tableau construction, *after* slack/surplus/artificial counts were decided. This meant the wrong variable types could be assigned to a flipped constraint.
- **After:** A dedicated pre-pass multiplies the entire constraint row by −1 and flips LE↔GE *before* any variable counting. This ensures the correct number and type of auxiliary variables.
- Simplified the tableau construction loop by removing the inline sign-flipping code.

---

## Commit 4 — `2ed73f0` — *Added step-by-step tableau*

**Rich, human-readable tableau printing for debugging and educational use:**

1. **`getVarLabel()`** — maps variable indices to labels: `x0`, `x1`, ..., `s0`, `s1`, ..., `a0`, `a1`, ...
2. **`printTableau()` rewrite** — now prints a formatted table with:
   - Column headers (variable labels + `RHS`).
   - A `Basis` column showing which variable is basic in each row.
   - Separator lines between the objective row and constraint rows.
3. **`printTableauStep()`** — prints iteration context: entering/leaving variable names and their indices.
4. **Integrated into `runSimplexPhase()`** — the initial tableau is printed before any pivoting, and the tableau is printed after every pivot operation. The old `printTableau` call from `main()` was removed.

---

## Commit 5 — `3ae2b9c` — *Added silent option*

**Introduced a global verbosity flag (`g_verbose`) and a `-s`/`--silent` CLI flag:**

- **`g_verbose`** (default 1) — all `printf` calls throughout the code are now gated behind `if (g_verbose)`.
- Affected areas: MPS parsing summary, bound preprocessing summary, tableau dimensions, iteration logs, phase announcements, CUDA device info, test problem description, step-by-step tableau prints.
- **CLI parsing** — `main()` now iterates over `argv` looking for `-s`/`--silent` (sets `g_verbose = 0`) or a positional filename argument, instead of just using `argv[1]`.
- This was essential for **automated benchmarking** (the `speed_test.py` script), where verbose output would pollute timing measurements.

---

## Commit 6 — `1dec1fc` — *Added compile emps and refactored*

**Small but impactful change:**

- Increased `maxIterations` from **1,000 to 10,000** — necessary for solving larger Netlib problems (like `bnl2`, `d2q06c`, `degen3`) that require many more simplex iterations.

---

## Commit 7 — `9fcb840` — *Changed times to hpc functions* (current HEAD)

**Added performance timing using HPC timing functions:**

- **`#include "hpc.h"`** — imported the HPC header for high-resolution timing.
- **Timing instrumentation** — wrapped `solveSimplex()` in `hpc_gettime()` calls, measuring only the solve phase (excluding I/O and parsing).
- **Elapsed time output** — prints `Elapsed time: X.XXXXXX seconds` at the end.

---

## Summary of Evolution

| Feature | Commit | Impact |
|---|---|---|
| Core CUDA simplex + basic MPS parser | `00d449e` | Foundation |
| Robust fixed-column MPS parser | `e7f573b` | Correctness — handles real Netlib files |
| Bounds/Ranges/Integer preprocessing | `e7f573b` | Completeness — standard LP features |
| Dynamic memory (no fixed limits) | `e7f573b` | Scalability |
| Single-pass MPS parsing | `e7f573b` | Performance |
| Negative RHS pre-pass fix | `a90a841` | Correctness — bug fix |
| Step-by-step tableau visualization | `2ed73f0` | Debugging/Education |
| Silent mode (`-s` flag) | `3ae2b9c` | Usability — enables scripted benchmarks |
| 10× iteration limit increase | `1dec1fc` | Scalability — larger problems |
| HPC timing instrumentation | `9fcb840` | Benchmarking |

The code matured from a basic prototype that could solve toy problems into a robust solver capable of handling standard Netlib benchmark instances, with proper MPS compliance, preprocessing, debugging output, and performance measurement.
