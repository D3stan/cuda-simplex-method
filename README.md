# CUDA Two-Phase Simplex Method Solver

A GPU-accelerated implementation of the Simplex algorithm using CUDA, supporting the two-phase method for solving linear programming problems.

## Features

- **Double precision arithmetic** for numerical stability
- **MPS file format input** - standard LP file format
- **Handles all constraint types**: `<=`, `>=`, and `=` constraints
- **Supports both maximization and minimization** problems
- **Two-Phase Simplex Method** for problems with artificial variables
- **GPU-accelerated kernels** for parallel pivot operations
- **Built-in test problem** when no input file is provided

## Architecture

The implementation follows the design from `implementazione.md` and is split into lightweight modules:

- `core/`: shared types, constants, and runtime state (`SolverConfig`, `RunContext`)
- `parser/`: MPS parsing + bounds/ranges preprocessing
- `solver/`: CUDA kernels, tableau management, two-phase simplex logic
- `io/`: text/JSON/CSV output and batch summaries
- `app/`: CLI, interactive mode, and execution flow

### CUDA Kernels

1. **Kernel A** (`kernelFindPivotColumnSimple`): Finds the entering variable using parallel reduction to find the most negative reduced cost.

2. **Kernel B** (`kernelFindPivotRow`): Performs the minimum ratio test in parallel to find the leaving variable. Implements Bland's anti-cycling rule for tie-breaking.

3. **Kernel C** (`kernelUpdateTableau`): Updates the entire tableau in parallel using a 2D thread grid. Pre-caches pivot row and column in device memory for coalesced access.

### Two-Phase Method

- **Phase 1**: Minimizes the sum of artificial variables to find a basic feasible solution
- **Phase 2**: Optimizes the original objective function using the feasible basis

## Building

### Requirements

- NVIDIA CUDA Toolkit (11.0+)
- C++ compiler compatible with CUDA (g++ on Linux, Visual Studio on Windows)

### Linux

```bash
nvcc -O3 simplex.cu app/app.cu parser/parser.cu solver/kernels.cu solver/solver.cu io/io.cu -o simplex
```

### Makefile (consigliato)

```bash
make
```

Comandi utili:

```bash
make clean
make run
```

### Windows (Visual Studio Developer Command Prompt)

```bash
nvcc -O3 simplex.cu app/app.cu parser/parser.cu solver/kernels.cu solver/solver.cu io/io.cu -o simplex.exe
```

### With explicit C++14 (if needed)

```bash
nvcc -O3 -std=c++14 simplex.cu app/app.cu parser/parser.cu solver/kernels.cu solver/solver.cu io/io.cu -o simplex
```

## Usage

### With MPS file input

```bash
./simplex problem.mps
```

### Without arguments (runs built-in test problem)

```bash
./simplex
```

## MPS File Format

The solver supports the standard MPS format:

```
NAME          ProblemName
OBJSENSE
 MAX
ROWS
 N  OBJ        ; N = objective, L = <=, G = >=, E = =
 L  C1
 G  C2
 E  C3
COLUMNS
    X1        OBJ       3   C1        1
    X1        C2        2
    X2        OBJ       2   C1        1
RHS
    RHS1      C1        4
    RHS1      C2        6
ENDATA
```

### Sections

- `NAME`: Problem name
- `OBJSENSE`: Optional, followed by `MAX` or `MIN` (default: MIN)
- `ROWS`: Constraint definitions
  - `N`: Objective function (free row)
  - `L`: Less than or equal (`<=`)
  - `G`: Greater than or equal (`>=`)
  - `E`: Equality (`=`)
- `COLUMNS`: Variable coefficients in constraints
- `RHS`: Right-hand side values
- `BOUNDS`: Variable bounds (not yet implemented)
- `ENDATA`: End of file marker

## Example Problems

### test_problem.mps (Simple LP)

```
Maximize: 3*x1 + 2*x2
Subject to:
    x1 + x2 <= 4
   2*x1 + x2 <= 6
Expected: x1=2, x2=2, z=10
```

### test_twophase.mps (Requires Two Phases)

```
Minimize: 2*x1 + 3*x2 + x3
Subject to:
   x1 + x2 + x3 = 10
   x1 + 2*x2 >= 6
       x2 + x3 <= 8
```

## Output

The solver outputs:

1. Problem information (variables, constraints, optimization sense)
2. Tableau dimensions and variable counts
3. Initial tableau (optional debug)
4. Iteration progress showing pivot operations
5. Final solution with variable values and objective value

### Status Codes

- **OPTIMAL**: Optimal solution found
- **INFEASIBLE**: No feasible solution exists (Phase 1 objective > 0)
- **UNBOUNDED**: Objective can be improved indefinitely
- **ERROR**: Maximum iterations reached or computational error

## Implementation Details

### Memory Layout

- **Flattened 1D array** for tableau data (row-major order) ensures coalesced memory access
- Separate arrays for basic variable indices
- Host and device copies maintained for synchronization

### Optimization Techniques

- **Shared memory** used in reduction kernels
- **Coalesced global memory access** in tableau updates
- **Pivot data caching** before tableau update kernel
- **Single-block kernels** for small-medium problems (reliable and simple)

### Numerical Considerations

- Epsilon tolerance: `1e-10` for comparisons
- Handles negative RHS by row negation
- Big-M method for Phase 2 (artificial variable costs = 1e10)

## Limitations

- Variable bounds (other than non-negativity) not yet implemented
- RANGES section not supported
- No sparse matrix representation (dense tableau)
- Single GPU only

## Performance Notes

For large-scale problems:
- The tableau update kernel benefits most from GPU parallelization
- Pivot finding uses parallel reduction but is limited by reduction overhead for small dimensions
- Memory transfer between host and device is minimized (only at start and end)

## License

MIT License
