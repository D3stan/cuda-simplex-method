## Pass 2 — Detailed Problem Analysis

This pass revisits every finding above and deepens the description of *why* it is a problem, what its observable consequences are, and how widely the damage propagates through the codebase.

---

### A. DRY Violation — Duplicated `hasFileExtensionIgnoreCase` - Fixed

**Detail**: The two copies live in modules with different responsibilities: `parser.cu` uses it to route to `parseDAT` vs `parseMPS`, while `app.cu` uses it for the interactive-mode file check and the batch directory filter (`isSupportedInputFile`). Because each copy is `static`, neither is visible to the other translation unit — there is no shared declaration and no linker-level duplication warning to catch a divergence. The duplication is therefore invisible to the build system.

The risk is latent but concrete: if the case-folding logic ever needs to be changed (e.g., to handle locale-specific characters, or to add a new extension globally), a developer must remember — or discover — that there are two places to edit. If only one is updated, `app.cu` and `parser.cu` will disagree on which filenames are valid, producing a class of bugs where the parser accepts a file that the app has already rejected as unsupported, or vice versa.

---

### B. Magic Number — `1e20` Big-M Constant - Fixed

**Detail**: The three sites where `1e20` appears are:
1. `setupPhase2` — setting non-basic artificial columns to a large cost after restoring the Phase 2 objective.
2. `rederiveObjectiveRow` — same semantic, same value, same purpose, different function.
3. The `bigval` variable in the unbounded-recovery branch of `runSimplexPhase` — again the same value, but this one is also written back to device memory via a `cudaMemcpy`.

There is no documentation at any of these sites that explains the choice of `1e20` relative to the `EPSILON` and `PIVOT_TOL` tolerances defined in `simplex_core.h`. The value implicitly needs to be large enough that no legitimate reduced cost can exceed it, but small enough to not cause overflow when participating in floating-point arithmetic with other tableau entries. That relationship is nowhere expressed. If `EPSILON` or `PIVOT_TOL` were to change — or if a problem with very large coefficients were introduced — the adequacy of `1e20` would have to be re-evaluated at each site independently rather than from a single definition.

---

### C. Magic Numbers — Hardcoded Buffer and Capacity Sizes - Fixed

**Detail**:

- `1024` (line buffer in `parseMPS` / REPL buffer in `interactiveMode`): the two uses have different purposes but the same value. An MPS file with a line longer than 1023 bytes will be silently truncated by `fgets`, potentially losing the second value on a data line and misparsing the following lines. The REPL buffer has different overflow semantics — a user command longer than 1023 characters gets split into two `fgets` reads and will be processed incorrectly as two commands.

- `512` (path buffer, discussed further under finding M).

- `4096` (batch file array): this is an upper bound on how many files the batch mode will process. If more than 4096 files are provided or discovered by directory expansion, `expandedFiles[expandedCount++]` writes past the end of the allocated array. There is no bounds check on `expandedCount` anywhere in the loop. The overflow is silent.

- `64`/`256` (initial `rowCap`/`varCap`/`coeffCap` in `parseMPS`): these are just starting capacities; the arrays are grown dynamically. The values themselves are not dangerous, but their arbitrary choice is not explained, and neither is the doubling strategy. Problems with more than 64 rows or 64 variables will trigger at least one reallocation on every parse.

---

### D. Magic Numbers — Hardcoded MPS Field Column Offsets - Fixed

**Detail**: The MPS fixed-column format offsets are used as raw integers in eight `extractMPSField` calls per section, and the same six-field pattern is repeated in four sections (COLUMNS, RHS, BOUNDS, RANGES). That is 24 call sites all containing the same six pairs of numbers.

The deeper problem is that the offsets are not uniformly applied: the BOUNDS section uses `1–2` for the bound type (field 1) and shifts name/value fields to `4–11`, `14–21`, `24–35`, while the other sections use `4–11` for a name and `14–21` for a second name starting at the same physical layout. This divergence is opaque when reading any single section's code because the intent of each integer pair is not self-documented. The MPS standard definition is not cited in the code, so a reader must externally verify that `24` means "start of the value field" and not "arbitrary offset chosen by the author."

---

### E. God Function — `parseMPS`

**Detail**: `parseMPS` conflates at least six distinct responsibilities:

1. **File I/O** — opening, reading line-by-line, closing.
2. **Section dispatch** — maintaining a `section` string and selecting a code path per section.
3. **Dynamic array management** — growing `rowNames`, `varNamesBuf`, `rhsValues`, `rangeVals`, `coeffs` with inline lambdas.
4. **Domain-specific parsing** — interpretting type characters (`L`, `G`, `E`, `N`), integer markers (`INTORG`/`INTEND`), bound types (`LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`, `LI`, `UI`), and `OBJSENSE`.
5. **Struct assembly** — converting temporary sparse data into a fully allocated `LPProblem`.
6. **Cleanup** — freeing all temporary allocations.

The use of C++ lambdas (`auto growVarArrays = [&]() { ... }`) to capture mutable state by reference is correct but adds a new complexity layer: any variable in the enclosing scope can be silently read or modified by any lambda. This means that a lambda call in the middle of the COLUMNS parsing section can alter the state of memory that the BOUNDS section will later access, with no explicit data-flow indication.

The error-handling pattern is also inconsistent within the function: some parse errors `fclose(file)` and `freeLPProblem(lp)` before returning NULL, while others simply `continue` silently, and the lambdas that can fail (`growVarArrays`, etc.) return `void` with no error reporting mechanism at all.

---

### F. God Function — `solveSimplex`

**Detail**: `solveSimplex` contains three structurally different sub-problems in sequence:

1. A Phase 1 / Phase 2 scheduling block (choosing whether Phase 1 is needed).
2. A 170-line numerical tableau rehabilitation block between phases — perturbation removal, B⁻¹ accuracy checking via iterative refinement, full column re-derivation — gated on the bitmask `extractionStatus & 2`.
3. Phase 2 cost vector construction and `setupPhase2` call.

The rehabilitation block is the most problematic: it allocates and frees four separate arrays (`initialBasisCol`, `varToConstraint`, `varCoeffSign`, `residual`, `isBasicVar`) and performs O(nCons²) work. This block is only conditionally executed based on a bitmask return value from `extractArtificialBasis`, but the condition (`(extractionStatus & 2) == 0`) is not documented with the meaning of each bit. A reader must trace backwards through `extractArtificialBasis` to understand when this branch executes.

The consequence is that `solveSimplex` is difficult to test in isolation: to exercise the rehabilitation branch requires a problem whose artificial basis extraction happens to return a specific bitmask value, and that condition is not exposed through any parameter.

---

### G. God Function — `runSimplexPhase`

**Detail**: The unbounded-recovery sub-procedure embedded within `runSimplexPhase` is itself a non-trivial algorithm. When `h_pivotRow < 0`, the function:

1. Attempts a host-side ratio test with a relaxed threshold (`HARRIS_TOL`).
2. If that fails, blocks the current column by writing `1e20` to both host and device memory.
3. Re-derives the objective row.
4. Re-writes `1e20` again to the now-derived column (because re-derivation overwrites it).
5. Launches `kernelFindPivotColumnSimple` again.
6. Launches `kernelFindPivotRow` with the new column.
7. Reads back two device values.
8. If still not recovered, restores the objective row via `rederiveObjectiveRow`.

This is approximately 60 lines of inline recovery logic that performs three conditional CUDA kernel launches inside the main loop's `if (h_pivotRow < 0)` block. The nesting depth at the innermost point reaches 5–6 levels. The recovery logic is structurally interleaved with the normal pivot path, making it impossible to read just the normal path or just the recovery path in isolation. The re-writing of `1e20` after `rederiveObjectiveRow` is particularly fragile — it relies on the call order and on the fact that `rederiveObjectiveRow` does not restore the column before the caller overwrites it again.

---

### H. God Function — `runApp`

**Detail**: `runApp` is both the CLI argument parser and the application entry point. The argument-parsing section alone spans roughly 80 lines, with a flat chain of `if/else if` comparisons. Each option that requires a subsequent argument (e.g., `--max-iter`, `--timeout`, `--log`) contains its own error message and early `return EXIT_FAILURE` with a `free(inputFiles)` call. If a future option is added that also needs cleanup of `expandedFiles` (which is allocated later), every one of these early-return sites would need to be updated, and the omission of any one of them would produce a leak.

The batch mode inner loop is ~100 lines long and resides directly inside `runApp`, not in a dedicated function. It contains a progress-reporting `fprintf(stderr, ...)` that is active only when `savedVerbose > 0`, where `savedVerbose` is a locally captured copy of `config.verbose` that was temporarily zeroed — a non-obvious two-state pattern. The final `fprintf(stderr, "\r                              \r")` that erases the progress line is a hardcoded 30-space string with no explanation of why 30 spaces are sufficient.

---

### I. Missing Error Handling — `realloc` Without NULL Check

**Detail**: The C standard specifies that `realloc` returns NULL on failure and that the original pointer remains valid and unchanged. The `growVarArrays` lambda performs:

```c
varNamesBuf   = (VarName*)realloc(varNamesBuf, varCap * sizeof(VarName));
objCoeffsTemp = (double*)realloc(objCoeffsTemp, varCap * sizeof(double));
loBounds      = (double*)realloc(loBounds, varCap * sizeof(double));
upBounds      = (double*)realloc(upBounds, varCap * sizeof(double));
isInt         = (int*)realloc(isInt, varCap * sizeof(int));
```

If the second `realloc` fails and returns NULL, `objCoeffsTemp` becomes NULL while `varNamesBuf` now points to the newly resized memory. The old `objCoeffsTemp` buffer is lost — leaked — and the function continues to index `objCoeffsTemp[numVars]` in `addVar`, producing a NULL dereference. The same structural problem exists in `growRowArrays` (four reallocations) and the coefficient array growth (one reallocation). Because the lambdas return `void`, there is no mechanism for the calling code to detect the failure.

---

### J. Missing Error Handling — `strdup` Without NULL Check

**Detail**: `strdup` calls `malloc` internally and returns NULL on allocation failure. In `parseMPS`, the two most critical callsites are:

1. `objRowName = strdup(nameField)` — if this returns NULL, then every subsequent `findRow` call that compares against `objRowName` executes `strcmp(name, NULL)`, which is undefined behavior.
2. `rowNames[numRows] = strdup(nameField)` — if this returns NULL, then during struct assembly (the `BUILD LPProblem` block), `lp->constraintNames[i] = rowNames[i]` assigns NULL to the struct, and any subsequent code that prints or compares constraint names will dereference NULL.

Neither of these are hypothetical: on any system under memory pressure, allocation failures are possible. Unlike `malloc` failures which can be visually spotted by looking for explicit `malloc` calls, `strdup` failures are easy to overlook because `strdup` looks like a string operation rather than an allocation.

---

### K. Memory Leak — Batch Mode Early Exit

**Detail**: The allocation sequence in `runApp` is:

1. `inputFiles = malloc(argc * sizeof(const char*))` — early in the function.
2. (argument parsing loop)
3. `expandedFiles = malloc(4096 * sizeof(const char*))` — inside `if (batchMode)`.
4. (directory expansion loop, which may additionally `malloc` individual `fullpath` strings)
5. `free(inputFiles); inputFiles = expandedFiles;` — at the end of the expansion.
6. CUDA device check.

If the CUDA check fails between steps 3 and 5, both `inputFiles` and `expandedFiles` exist in memory. At the failure return site, the code frees `(void*)inputFiles` — but `expandedFiles` is a separate allocation that is never freed. Additionally, any `fullpath` strings that were individually `malloc`'d during directory scanning (step 4) are elements inside `expandedFiles` and are also leaked.

---

### L. Memory Leak — `realloc` Failure in `parseMPS`

**Detail**: `parseMPS` allocates the following temporary arrays before the parse loop:
`rowNames`, `rowTypes`, `rhsValues`, `rangeVals`, `varNamesBuf`, `objCoeffsTemp`, `loBounds`, `upBounds`, `isInt`, `coeffs`.

At the bottom of the function, all of these are freed explicitly. However, there is no error-recovery path that reaches this cleanup block if an intermediate failure occurs. The function has no `goto cleanup` pattern, no RAII, and no wrapper that would ensure cleanup. If `realloc` inside `growVarArrays` returns NULL and the caller does not check (as established in finding I), the function will crash before reaching the cleanup — leaving all of the above buffers leaked. If, hypothetically, the crash were avoided by a NULL check that returned early, the same leak would occur because the early-return path also does not free these buffers.

---

### M. Buffer Overflow Risk — Path Assembly in Batch Directory Expansion

**Detail**: The vulnerability is:

```c
char* fullpath = (char*)malloc(512);
snprintf(fullpath, 512, "%s/%s", inputFiles[i], entry->d_name);
```

`inputFiles[i]` is a raw command-line argument with no length validation. `entry->d_name` is a filename from the operating system, which on Linux can be up to 255 bytes (NAME_MAX). Even ignoring pathological filenames, a user providing a directory path longer than 255 bytes — entirely within normal operating system limits — would overflow a 512-byte buffer. `snprintf` writes at most 511 bytes and null-terminates, so there is no stack smash, but the resulting path is silently truncated. A truncated path refers to a non-existent or unintended file, which will then either fail to open (producing a misleading parse error) or — in a filesystem tree where a truncated path coincidentally exists — open the wrong file entirely. The silent nature of `snprintf` truncation means no warning is produced and the error propagates downstream.

---

### N. Naming Inconsistency — Three Names for the Same Concept

**Detail**: The number of constraints in an LP problem is:
- `lp->numConstraints` in `LPProblem` (defined in `simplex_core.h` and used in `parser.cu`, `solver.cu`, `io.cu`)
- `numRows` as a local variable in `parseMPS` and `parseDAT` (because during parsing the MPS row list has not yet been identified as a constraint list)
- `nCons` as a local variable in `solveSimplex` (line ~920 and elsewhere in the Phase 2 preparation block)

The divergence between `numRows` and `numConstraints` is a logical consequence of MPS terminology (the format calls them "rows"), and that is defensible inside the parser. However, `nCons` inside `solveSimplex` is a pure abbreviation introduced for no structural reason — `tab->rows - 1` also computes the same value and is used interchangeably with `nCons` in the same function. The result is that within `solveSimplex`, a reader encounters three expressions that all mean the same number: `lp->numConstraints`, `nCons`, and `tab->rows - 1`.

---

### O. `strcpy` for Known-Length Constants

**Detail**: The five consecutive `strcpy` calls to assign section names in `parseMPS`:

```c
if (strncmp(rawLine, "ROWS", 4) == 0)     { strcpy(section, "ROWS"); continue; }
if (strncmp(rawLine, "COLUMNS", 7) == 0)  { strcpy(section, "COLUMNS"); continue; }
...
```

The `section` buffer is declared as `char section[64]`. The longest section name is `"COLUMNS"` (7 characters). There is no overflow risk here today. The problem is that the same section names are then used for comparison with `strcmp(section, "ROWS")`, `strcmp(section, "COLUMNS")`, etc., throughout the data-line dispatch block. The name literal `"ROWS"` therefore appears in the file twice: once as the source string for `strcpy` (section detection) and once as the target string for `strcmp` (data-line dispatch). If a section name were ever misspelled in the `strcpy` call, data lines for that section would be silently skipped with no warning — `section` would hold the misspelled value, no `strcmp` would match it, and the data would fall into the "Unknown section: silently skip" path.

---

### P. Macro-Aliased Config/Run Fields

**Detail**: The eight macros defined at the top of `solver.cu`:

```c
#define g_verbose       (config->verbose)
#define g_outputFormat  (config->outputFormat)
#define g_iterLog       (config->iterLog)
#define g_debug         (config->debug)
#define g_maxIter       (config->maxIter)
#define g_timeout       (config->timeout)
#define g_phase         (run->phase)
#define g_totalIterations (run->totalIterations)
#define g_solveStartTime  (run->solveStartTime)
```

These macros turn struct member accesses into what appear to be global variables. The problem is three-fold:

1. **Scope opacity**: `g_verbose` at a glance looks like a file-scope global. It is actually a dereference of `config`, a parameter of the *calling* function, not of the function where the macro appears. This is not apparent without reading the macro definition.

2. **Mutation disguise**: `g_phase = 1` and `g_totalIterations++` look like assignments to globals. They are actually writing through the `run` pointer to the caller's `RunContext`. The caller (e.g., `runApp`) does not expect `solveSimplex` to directly modify `run->phase` via a macro alias — but it does.

3. **Debugger/tool confusion**: any debugger, profiler, or static analyser that does not perform macro expansion will report references to undefined variables. A `grep` for `config->verbose` will not find its uses inside `solver.cu`, making code search unreliable.
