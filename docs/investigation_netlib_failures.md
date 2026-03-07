# Investigation Plan: Netlib Failure Cases

## Overview

Five Netlib benchmark problems fail to produce the correct result.
Each exhibits a distinct failure mode at a different layer of the solver:

| Problem   | Rows  | Cols  | Artificials | Failure Mode                                    | Expected Optimal |
|-----------|-------|-------|-------------|------------------------------------------------|-----------------|
| 25fv47    | 821   | 1571  | 520         | **False INFEASIBLE** — Phase 1 obj = 164.3     | 5.5018E+03      |
| degen3    | 1503  | 1818  | ~1503       | **Failed artificial extraction** — rhs 5.07 in basis | −9.8729E+02 |
| sctap3    | 1480  | 2480  | 1480        | **Phase 1 spurious UNBOUNDED** — col −1 pivot  | 1.4240E+03      |
| bnl2      | 2324  | 3489  | 1716        | **Phase 1/2 error** after ~192 s               | 1.8112E+03      |
| d2q06c    | 2172  | 5167  | unknown     | **Timeout / no convergence** > 60 s            | 1.2278E+05      |

The failures are listed from most-likely-to-be-a-code-bug (25fv47, sctap3) to
most-likely-to-be-numerical-drift (bnl2, d2q06c). Investigate in this order.

---

## Problem 1 — 25fv47: False INFEASIBLE (Suspected Parser/Tableau-Setup Bug)

### Observed Behaviour

Phase 1 terminates at iteration 3361 with:
```
Phase 1 objective value: 164.3073680901 (tolerance: 6.756840e+01)
Problem is INFEASIBLE (Phase 1 objective = 164.307368)
```
The true optimal is 5501.8 — the problem is **feasible**.

### Root Cause Hypothesis

25fv47 contains:
- 516 **E** (equality) constraints — 5 of which have **negative RHS** (e.g. `RE060 = −4.7`, `RH009 = −1.0`, `RH016 = −14.2`, `2RH025 = −16.0`, `R1043 = −0.6`)
- 305 **L** constraints — 4 of which have **negative RHS** (e.g. `RCRFT = −62.5`, `1RE061 = −160.0`, `R1042 = −10.0`, `2RJ037 = −2.0`)

Standard practice: when an **E** row has negative RHS $b_i < 0$, negate the
entire row so $b_i > 0$ before placing an artificial variable. Similarly, an
**L** row with negative RHS is equivalent to a **G** row after negation (it then
needs a surplus variable and an artificial).

**Hypothesis**: the parser or tableau-builder does not negate E-rows (or L-rows)
with negative RHS, leaving artificials with an infeasible initial value equal to
$|b_i|$. The Phase 1 objective starts at the sum of these magnitudes
($\approx 4.7 + 1.0 + 14.2 + 16.0 + 0.6 + 62.5 + 160.0 + 10.0 + 2.0 = 271.0$)
and simplex can only reduce it partially — the residual 164.3 is the un-driven
portion — so the solver concludes INFEASIBLE.

The discrepancy between 520 artificial variables (solver-reported) and 516 E
rows further supports this: 4 extra artificials were added for the 4 negative-RHS
L rows (treated as G after negation), confirming partial handling. The negation
of E-rows is the missing step.

### Investigation Steps

**Step P1-A — Confirm the initialisation values of artificial variables**

Add a temporary diagnostic print immediately after `createTableau` (in
`solver.cu` or `app.cu`) that dumps, for each artificial variable column, its
tableau row index and the RHS value of that row:

```c
for (int r = 0; r < tab->rows - 1; r++) {
    if (tab->basis[r] >= tab->numOriginal + tab->numSlack + tab->numSurplus)
        printf("[DIAG] art basis row %d  rhs=%.6f\n", r, tab->rhs[r]);
}
```

If any of these RHS values is large and positive (e.g. 160.0, 62.5) at
iteration 0 — before any pivot — the tableau initialisation is incorrect.

**Step P1-B — Trace the negative-RHS E rows through the parser**

In `parser/parser.cu`, locate where **E** constraints are converted to tableau rows.
Check whether the code path that flips the sign of the row (and its RHS) when
`rhs[i] < 0` applies to E rows as well as G rows. A G row with `rhs < 0` becomes
`−Ax ≤ −b` (i.e. an L row); an E row with `rhs < 0` must also be negated so the
artificial starts at a non-negative value.

**Step P1-C — Check the SCSD8 discrepancy as a calibration**

SCSD8 is OPTIMAL but reports 906.2 vs reference 905.0.
After fixing 25fv47's parser issue, re-run SCSD8.
If the discrepancy disappears, it was caused by the same sign-handling error
affecting a subset of its constraints.

### Expected Fix

In the tableau builder, for every constraint row $i$:
```
if (rhs[i] < 0) {
    negate row i of A;
    negate rhs[i];
    if type was L → change to G (add surplus + artificial);
    if type was E → keep as E but artificial starts at the (now positive) rhs[i];
}
```

---

## Problem 2 — sctap3: Spurious Phase 1 UNBOUNDED (Suspected Control-Flow Bug)

### Observed Behaviour

Phase 1 terminates at iteration 3504 with:
```
Iteration 3504: Problem is unbounded (no valid pivot row for column -1)
Error: Phase 1 should not be unbounded!
```

The pivot column reported is **−1**, which is not a valid column index.

### Root Cause Hypothesis

The entering-variable selection kernel returns a sentinel value (e.g. −1) when
no eligible column is found — the intended meaning is "optimality" or
"all reduced costs ≥ 0". The control flow in `runSimplexPhase` then passes
this −1 directly to the ratio-test kernel, which finds no positive entry in a
non-existent column and triggers the UNBOUNDED path instead of the OPTIMAL path.

In other words: an **early-exit optimality signal is misrouted to the unbounded
handler** because the −1 sentinel check is missing (or is ordered after the
unbounded check).

### Investigation Steps

**Step S2-A — Locate the sentinel check in `runSimplexPhase`**

In `solver.cu`, find the section after `selectPivotColumn` (or equivalent GPU
kernel launch) where the returned column index is checked. It should be:

```c
if (pivotCol < 0) {
    // OPTIMAL — all reduced costs ≥ 0
    break;
}
```

If this check is absent, or is placed *after* the ratio-test call, the −1 is
forwarded and the ratio-test correctly finds "no valid row" — triggering UNBOUNDED.

**Step S2-B — Collect the health log**

```bash
mkdir -p investigation/sctap3
./simplex.out netlib/sctap3.mps \
    --log investigation/sctap3/iters.csv \
    --health-log investigation/sctap3/health.csv
```

Examine `health.csv`. Focus on:
- `neg_rhs_count` near iteration 3500: if it is 0, Phase 1 was genuinely on track and the −1 col is the only cause.
- `min_reduced_cost` in the last few rows: if it is ≥ 0, this confirms the solver *did* find Phase 1 optimum and the −1 col is a correct "no entering variable" signal.

**Step S2-C — Inspect the last 10 iterations in `iters.csv`**

```bash
awk -F, 'NR==1 || $1 >= 3495' investigation/sctap3/iters.csv
```

Look at the `reduced_cost` and `ratio` columns. If `reduced_cost` ≥ 0 for the
last pivot and the next row has `pivot_col = −1`, the OPTIMAL signal was raised
but not caught.

### Expected Fix

Add (or reorder) the `pivotCol < 0` sentinel check:
```c
int pivotCol = selectEnteringVariable(tab, config);
if (pivotCol < 0) {
    status = SIMPLEX_OPTIMAL;
    break;  // do NOT fall through to ratio test
}
```

---

## Problem 3 — degen3: Artificial Variable Not Driven to Zero (Degenerate Phase 1)

### Observed Behaviour

Phase 1 terminates at iteration 5946 with apparent optimality:
```
Phase 1 objective value: -12.2900000000 (tolerance: 2.262016e+02)
```
The tolerance (226.2) far exceeds the objective magnitude (12.29), so the
feasibility check passes. Then extraction fails:
```
Error: Artificial variable in basis has non-zero value (row 1281, rhs 5.069200e+00)
Error: Failed to extract degenerate artificial basis
```

The Phase 1 objective **is negative** — the sum of artificial variables cannot
be negative, so a value of −12.29 is numerically impossible. This is numerical
drift: the objective row accumulated enough error that it reports −12.29 when
the true value is some small positive number. The feasibility pass is therefore
a false negative, and there are one or more artificials still nonzero in the
basis (the largest having value 5.069).

### Root Cause Hypothesis

DEGEN3 has 1504 rows and takes ~5946 Phase 1 iterations. By this point, the
tableau has undergone thousands of pivots without constraint-column
refactorisation. The objective row is re-derived every 50 iterations but reads
from already-drifted constraint rows, propagating the accumulated error into the
reduced costs. The large dynamic range of DEGEN3's coefficients (the problem is
a degenerate LP designed to stress ratio-test anti-cycling) exacerbates drift.

The large tolerance value (226.2) is computed from the current tableau entries
and is itself inflated by the drift — the tolerance that *should* catch this
residual is simply swamped.

### Investigation Steps

**Step D3-A — Collect health and iteration logs**

```bash
mkdir -p investigation/degen3
./simplex.out netlib/degen3.mps \
    --log investigation/degen3/iters.csv \
    --health-log investigation/degen3/health.csv
```

Expected: ~5946 rows in `iters.csv`, ~120 rows in `health.csv`.

**Step D3-B — Scan the health log**

In `health.csv`, look for:
- `neg_rhs_count > 0`: the first row with a positive value marks the onset of
  constraint-row corruption. In a valid Phase 1 BFS, all $b_i \geq 0$ is an
  invariant.
- `obj_rhs` turning negative: Phase 1 objective (sum of artificials) cannot be
  negative; a negative `obj_rhs` confirms the objective row has absorbed drift.
- `max_abs_entry` growing: a sudden jump from ~$10^3$ to $10^6$ or higher
  indicates blow-up in the constraint matrix.

**Step D3-C — Check Phase 1 tolerance computation**

In `solver.cu`, locate the Phase 1 feasibility check:
```c
double phase1Tol = ...;
if (fabs(phase1Obj) > phase1Tol) { INFEASIBLE; }
```
Verify how `phase1Tol` is derived. If it scales with `max_abs_entry` (as a
relative tolerance), a drifted tableau will produce an inflated tolerance that
masks a nonzero Phase 1 objective.

**Step D3-D — Verify the artificial-extraction fallback**

`extractArtificialBasis` at row 1281 finds rhs = 5.069 > 0. This means artificial
variable $a_{1281}$ is still basic with a positive value. Check whether the
extraction code attempts to pivot an artificial out even when its column is
numerically near zero, and whether it has a fallback for truly degenerate cases
(all entries in the artificial's row are below `PIVOT_TOL`).

### Expected Fix

Two complementary fixes:
1. **Lower `REFACTOR_COL_INTERVAL`** (currently 500, effectively disabled in Phase 1).
   Set it to ~200 for Phase 1 runs with many artificials. This resets $B^{-1}$
   from the original data and prevents drift accumulation.
2. **Tighten the Phase 1 feasibility tolerance**: use an absolute tolerance
   (e.g. `1e-4 * numArtificials`) rather than one derived from the drifted tableau.

---

## Problem 4 — bnl2: Error After ~192 Seconds (Drift Over Many Iterations)

### Observed Behaviour

```
Status: ERROR
Elapsed time: 192.628417 seconds
```

No error message was captured before termination (the run was cut off), but the
192-second runtime and large problem size (2324 rows, 3489 variables, 1716
artificials) point to the same drift mechanism seen in lp11.dat, amplified by
scale. The solver either cycles, accumulates enough numerical error to trigger an
assertion, or hits the maximum-iteration limit with an internal error state.

### Root Cause Hypothesis

Over tens of thousands of Phase 1 or Phase 2 iterations on a 2325 × 6203 tableau,
floating-point errors in $B^{-1}$ accumulate beyond the correction provided by
the 50-iteration objective-row re-derivation. With `REFACTOR_COL_INTERVAL = 500`,
full constraint-column refactorisation happens every 500 iterations at best,
which may be insufficient for a tableau of this size.

### Investigation Steps

**Step B4-A — Collect health and iteration logs**

```bash
mkdir -p investigation/bnl2
./simplex.out netlib/bnl2.mps \
    --log investigation/bnl2/iters.csv \
    --health-log investigation/bnl2/health.csv
```

This will run for ~192 seconds. The iteration log may be large (hundreds of
thousands of rows); use `wc -l` to gauge its size before opening it.

**Step B4-B — Identify the phase and iteration of failure**

```bash
tail -5 investigation/bnl2/iters.csv
tail -5 investigation/bnl2/health.csv
```

Determine whether the error occurs in Phase 1 or Phase 2, and at what
iteration count.

**Step B4-C — Scan the health log for the instability onset**

Apply the same analysis as for degen3 (Step D3-B):
- First `neg_rhs_count > 0` row → earliest visible corruption
- First `obj_rhs` decrement → objective row poisoned
- `max_abs_entry` trajectory

Derive the instability window `[W_start, W_end]` (50-iteration block).

**Step B4-D — Drill into the instability window**

```bash
awk -F, -v s=$((W_start-10)) -v e=$((W_start+50)) \
    'NR==1 || ($1>=s && $1<=e)' investigation/bnl2/iters.csv
```

Look for degenerate pivots (ratio ≈ 0), oscillating `obj_rhs`, or a sudden
change in pivot column patterns.

**Step B4-E — Test reduced `REFACTOR_COL_INTERVAL`**

Edit `core/simplex_core.h`:
```c
#define REFACTOR_COL_INTERVAL 200   // was 500
```
Rebuild and rerun. If the error disappears or the solver reaches an answer,
constraint-column drift was the primary cause. Try 100 as well if 200 still fails.

---

## Problem 5 — d2q06c: No Convergence / Timeout

### Observed Behaviour

The solver runs continuously for > 60 seconds without producing a result.
The problem has 2172 constraints and 5167 variables (the largest in the Netlib
set tested), with reference optimal 1.2278E+05.

### Root Cause Hypothesis

Two candidates:

1. **Cycling in Phase 1 or Phase 2**: the perturbation/anti-cycling mechanism
   (`PERTURB_EPS`, `PHASE2_PERTURB_EPS`) is overwhelmed by the problem's
   degeneracy, causing the solver to revisit the same basis repeatedly.

2. **Numerical drift causing non-termination**: similar to the lp11 / bnl2 case
   but severe enough that the solver never reaches a termination condition; no
   optimality or unboundedness is detected because the objective row is corrupted
   on both sides simultaneously.

### Investigation Steps

**Step D5-A — Profile iteration count vs. time with a short timeout**

```bash
mkdir -p investigation/d2q06c
timeout 30 ./simplex.out netlib/d2q06c.mps \
    --health-log investigation/d2q06c/health.csv 2>&1 | tail -5
wc -l investigation/d2q06c/health.csv
```

From the number of health rows (each = 50 iterations), estimate total iterations
in 30 seconds. If iterations/second is low (< 500/s), the problem may be that
each iteration is extremely expensive at this tableau size; the solver may simply
need much longer. If iterations/second is high but convergence is flat, it is
cycling.

**Step D5-B — Detect cycling in the health log**

```bash
awk -F, '{print $5}' investigation/d2q06c/health.csv | sort -n | uniq -c | sort -rn | head
```

Column 5 is `obj_rhs`. If the same objective value appears many times, the
solver is cycling.

**Step D5-C — Check the objective trajectory**

Plot or scan `obj_rhs` over time. For a valid Phase 2 minimization, it should be
monotonically non-decreasing (since the internal representation stores −z). Any
oscillation confirms cycling or objective-row corruption.

**Step D5-D — Test increased perturbation**

In `core/simplex_core.h`:
```c
#define PERTURB_EPS       1e-3   // was 1e-4
#define PHASE2_PERTURB_EPS 1e-3  // was 1e-4
```
Rebuild and rerun with a 60-second timeout. Stronger perturbation reduces cycling
at the cost of slightly suboptimal intermediate solutions.

**Step D5-E — Test reduced `REFACTOR_COL_INTERVAL`**

As for bnl2, lower `REFACTOR_COL_INTERVAL` to 200 or 100 and rerun. If the
solver now terminates (even slowly), constraint drift was the primary blocker.

---

## Summary: Root Cause Classification and Recommended Fix Order

| Problem | Root Cause Class                             | Primary Fix                                          |
|---------|----------------------------------------------|------------------------------------------------------|
| 25fv47  | **Parser/tableau-setup bug** (negative-RHS row negation missing for E rows) | Negate E rows with `rhs < 0` in the tableau builder |
| sctap3  | **Control-flow bug** (−1 sentinel not caught before ratio test) | Add `pivotCol < 0 → OPTIMAL` guard before ratio test |
| degen3  | **Numerical drift + tolerance mis-calibration** in Phase 1 | Lower `REFACTOR_COL_INTERVAL`; absolute Phase 1 tolerance |
| bnl2    | **Numerical drift** over many Phase 1/2 iterations | Lower `REFACTOR_COL_INTERVAL` to 200 |
| d2q06c  | **Cycling or drift** at large scale           | Increase `PERTURB_EPS`; lower `REFACTOR_COL_INTERVAL` |

Fix 25fv47 and sctap3 first — they are likely single-line code changes.
Fix degen3 and bnl2 together since they share the same `REFACTOR_COL_INTERVAL`
lever. Address d2q06c last; it may require multiple tuning iterations.
