# Investigation Plan: False UNBOUNDED on lp11.dat

## Problem Summary

`lp11.dat` (800 variables, 170 constraints, all GE ≥, all-positive objective, all-positive RHS)
terminates with **UNBOUNDED** after 11,649 iterations in Phase 2.
The expected result is **OPTIMAL** with objective value ≈ 1402.70.

Since all $c_j > 0$ and all $x_j \geq 0$, the problem is provably bounded below.
The UNBOUNDED result is a **solver error**, not a problem property.

## Working Hypothesis

After several thousand simplex pivots, floating-point errors accumulate in the
in-memory $B^{-1}$ representation (the tableau constraint-row entries). The
objective row is re-derived every 50 iterations (`REFACTOR_INTERVAL`), but it
reads from the already-drifted constraint rows — so reduced costs absorb the
same error. Eventually, at least one non-basic variable with a true reduced cost
$\geq 0$ appears negative (spurious entering variable). The subsequent ratio test
finds no positive entry in that column (also corrupted), and the solver declares
UNBOUNDED.

Key constants involved:
- `REFACTOR_INTERVAL = 50` — objective row periodically re-derived from tableau
- `REFACTOR_COL_INTERVAL = 999999` — constraint column re-derivation effectively disabled

---

## Tools Available

| Tool | What it produces |
|---|---|
| `--log <file>` | One CSV row **per iteration**: `iter, phase, pivot_col, pivot_row, reduced_cost, ratio, obj_rhs` |
| `--health-log <file>` | One CSV row **every 50 iterations** (each refactorization interval): `iter, phase, neg_rhs_count, max_abs_entry, obj_rhs, min_reduced_cost` — plus a final row at UNBOUNDED detection |
| `--diag` | Verbose diagnostic prints to stdout including negative-RHS counts and pivot column analysis at failure |

The health log is the primary tool for this investigation; it produces ~235 rows
(11,649 / 50) which is easy to scan. The iteration log (11,649 rows) is used
to drill into specific windows once the health log narrows the range.

---

## Step 0 — Baseline Run: Collect Logs

Create a working directory for this investigation and run once to collect both logs.

```bash
mkdir -p investigation/lp11

./simplex.out Dati-LP/lp11.dat \
    --log investigation/lp11/iters.csv \
    --health-log investigation/lp11/health.csv \
    -s
```

Runtime is approximately 1.4 seconds. Expected output:
- `iters.csv` : ~11,649 data rows
- `health.csv` : ~235 data rows + 1 final UNBOUNDED snapshot

---

## Step 1 — Scan the Health Log for Instability Onset

Open `health.csv` and look for these **signals**, column by column:

### 1a. `neg_rhs_count` (column 3)
Under correct operation this should always be 0 in Phase 2 — every RHS value
$b_i \geq 0$ is an invariant of a valid BFS. The **first row where `neg_rhs_count > 0`
is the earliest sign of numerical corruption** in the constraint rows.

### 1b. `max_abs_entry` (column 4)
Numerical blow-up often precedes visible sign errors. A sudden jump (e.g.,
from ~$10^3$ to $10^8$ in one 50-iteration window) is a red flag even before
`neg_rhs_count` turns positive.

### 1c. `obj_rhs` (column 5)
This is the current objective value ($-z$). For a minimization problem with
all-positive costs it should be **monotonically non-decreasing** in Phase 2.
Any decrease, or a large sudden jump, confirms the objective row has absorbed
numerical noise.

### 1d. `min_reduced_cost` (column 6)
Should converge toward 0 as the optimum is approached. If it oscillates or
suddenly becomes very negative late in the solve (e.g., at 10,000+ iterations),
that flags a spurious entering variable.

**Goal of Step 1:** Identify the *50-iteration window* (e.g., "between iteration
8,500 and 8,550") where corruption first appears. Note this window as `[W_start, W_end]`.

---

## Step 2 — Manual Inspection of Iteration Batches

Use `iters.csv` for targeted batch inspection. Each row has:
`iter, phase, pivot_col, pivot_row, reduced_cost, ratio, obj_rhs`

### Batch A — Early iterations (rows 10–14)
Verify the solver is behaving normally at the start. Reduced costs should be
negative (valid entering variables), ratios positive and well-conditioned.

```bash
awk -F, 'NR==1 || ($1>=10 && $1<=14)' investigation/lp11/iters.csv
```

### Batch B — Just before the instability window from Step 1 (rows W_start−10 to W_start−1)
The last "healthy" iterations before drift becomes visible. Check that `obj_rhs`
is still increasing and ratios are normal.

```bash
awk -F, -v s=$((W_start-10)) -v e=$((W_start-1)) \
    'NR==1 || ($1>=s && $1<=e)' investigation/lp11/iters.csv
```

### Batch C — Inside the instability window (rows W_start to W_start+9)
Look for the **first abnormal event**, such as:
- `ratio` becoming 0 or very small (degenerate pivots accumulating)
- `obj_rhs` decreasing (objective row corrupted)
- A sudden change in `pivot_col` pattern

### Batch D — Final 10 iterations before UNBOUNDED (rows 11,640–11,649)
```bash
awk -F, 'NR==1 || $1>=11640' investigation/lp11/iters.csv
```
Verify: `reduced_cost` is negative (triggers entering variable search), but
`pivot_row` should show −1 or nonsensical values confirming the failed ratio test.

---

## Step 3 — Correlate Objective Drift with Refactorization Intervals

Every 50 iterations the objective row is re-derived from the constraint rows.
If the constraint rows are already drifted, re-derivation propagates the error
into the objective row rather than correcting it.

From the health log, tabulate or plot:
- `obj_rhs` vs. `iter` — should be monotone; a kink or reversal marks danger
- `min_reduced_cost` vs. `iter` — should approach 0; unexpected negative dips
  after a re-derivation checkpoint (multiples of 50) implicate the re-derivation itself

**Key question to answer:** Does `min_reduced_cost` get more negative *right after*
a re-derivation checkpoint (at iter = 50k), or does it drift slowly between them?
- **More negative right after**: the objective re-derivation is reading dirty
  constraint columns and amplifying the error.
- **Gradual drift between checkpoints**: the pivot operations themselves are
  causing drift; re-derivation provides temporary relief but not enough.

---

## Step 4 — Verify Drift is in the Constraint Columns

A cross-check: if constraint columns are corrupted, the identity structure of
the basic-variable columns should be broken. At the final iteration (captured
in the health snapshot at UNBOUNDED detection), check the `max_abs_entry`
value. For a well-maintained tableau, basic-variable columns are identity columns
(entries 0 or 1) and non-basic entries are $B^{-1}N$ products; typical values
for this problem should be in the range of the original coefficient magnitudes
(~$10^0$–$10^3$). If `max_abs_entry` is $\gg 10^3$, that confirms the constraint
matrix has experienced numerical blow-up.

---

## Expected Outcome

The investigation should conclude with one of these diagnoses:

1. **Constraint column drift dominates**: `max_abs_entry` grows continuously,
   `neg_rhs_count` appears relatively early in Phase 2. Fix: lower
   `REFACTOR_COL_INTERVAL` (e.g., to 500) so `refactorConstraintColumns` runs
   periodically and resets $B^{-1}$ from the original LP data.

2. **Objective re-derivation amplifies existing error**: `min_reduced_cost`
   worsens right at re-derivation checkpoints despite `neg_rhs_count` remaining
   zero until late. Fix: re-derive the objective row less aggressively (increase
   `REFACTOR_INTERVAL`) or ensure the constraint columns are clean before each
   re-derivation.

3. **Combination**: both mechanisms are at play. Lowering `REFACTOR_COL_INTERVAL`
   is still the primary lever, since it addresses the root (constraint column
   drift) rather than the symptom (bad reduced costs).
