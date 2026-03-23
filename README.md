# CUDA Simplex Method

CUDA implementation of a two-phase simplex solver for linear programming problems.

## Purpose

This project solves linear programs on the GPU using the simplex method. It is meant for experimenting with CUDA-based optimization and comparing GPU execution against CPU solvers on benchmark datasets.

## How It Works

- Input: `.mps` or `.dat` linear programming problem files
- Processing: the parser loads the problem, builds the simplex tableau, and the solver runs the two-phase simplex method on the GPU
- Output:
  - default text output with solver status, objective value, elapsed time, and total iterations
  - optional JSON output with `--json`
  - optional CSV output with `--csv`

Typical statuses are `OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, `TIMEOUT`, and `ERROR`.

## Requirements

For local testing you need:

- an NVIDIA GPU
- CUDA Toolkit with `nvcc`
- `make`
- Python 3 for the benchmark scripts
- `highspy` if you want to run the CPU vs GPU comparison script

Install the optional Python dependency with:

```bash
pip install highspy
```

## How To Run

Build the solver:

```bash
make
```

Run a single problem:

```bash
./simplex.out data/netlib/afiro.mps
```

Run with JSON output:

```bash
./simplex.out --json data/netlib/afiro.mps
```

Run a batch of problems:

```bash
./simplex.out --batch data/netlib
```

Run the speed benchmark:

```bash
python3 scripts/speed_test.py
```
