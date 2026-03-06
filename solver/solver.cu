#include "solver.h"

extern __global__ void kernelFindPivotColumn(const double*, int, int, double*, int*);
extern __global__ void kernelFindPivotColumnSimple(const double*, int, double*, int*);
extern __global__ void kernelFindPivotRow(const double*, int, int, int, double*, int*);
extern __global__ void kernelUpdateTableau(double*, int, int, int, int, const double*, const double*);
extern __global__ void kernelCachePivotData(const double*, int, int, int, int, double*, double*);


// ===========================================================================

/**
 * Create and initialize the tableau from LP problem
 * Handles slack, surplus, and artificial variables
 */
Tableau* createTableau(LPProblem* lp, const SolverConfig* config) {
    Tableau* tab = (Tableau*)malloc(sizeof(Tableau));
    
    // Pre-pass: flip constraints with negative RHS so that b >= 0
    // This must happen BEFORE counting slack/surplus/artificial variables,
    // because flipping LE->GE (or GE->LE) changes the variable types needed.
    for (int i = 0; i < lp->numConstraints; i++) {
        if (lp->rhs[i] < 0) {
            // Multiply entire constraint by -1
            for (int j = 0; j < lp->numVars; j++) {
                lp->constraintMatrix[i][j] *= -1.0;
            }
            lp->rhs[i] *= -1.0;
            // Flip constraint type
            if (lp->constraintTypes[i] == CONSTRAINT_LE) {
                lp->constraintTypes[i] = CONSTRAINT_GE;
            } else if (lp->constraintTypes[i] == CONSTRAINT_GE) {
                lp->constraintTypes[i] = CONSTRAINT_LE;
            }
            // EQ stays EQ (just negated coefficients and RHS)
        }
    }
    
    // Count additional variables needed (after RHS flipping)
    int numSlack = 0;
    int numSurplus = 0;
    int numArtificial = 0;
    
    for (int i = 0; i < lp->numConstraints; i++) {
        switch (lp->constraintTypes[i]) {
            case CONSTRAINT_LE:
                numSlack++;
                break;
            case CONSTRAINT_GE:
                numSurplus++;
                numArtificial++;
                break;
            case CONSTRAINT_EQ:
                numArtificial++;
                break;
        }
    }
    
    tab->numOriginalVars = lp->numVars;
    tab->numSlack = numSlack + numSurplus;  // Both slack and surplus
    tab->numArtificial = numArtificial;
    
    // Tableau dimensions
    // Rows: 1 (objective) + numConstraints
    // Cols: original vars + slack/surplus + artificial + RHS
    tab->rows = lp->numConstraints + 1;
    tab->cols = lp->numVars + numSlack + numSurplus + numArtificial + 1;
    
    if (config->verbose) {
        printf("Tableau dimensions: %d x %d\n", tab->rows, tab->cols);
        printf("  Original variables: %d\n", lp->numVars);
        printf("  Slack variables: %d\n", numSlack);
        printf("  Surplus variables: %d\n", numSurplus);
        printf("  Artificial variables: %d\n", numArtificial);
    }
    
    // Allocate host memory
    size_t tableauSize = tab->rows * tab->cols * sizeof(double);
    tab->hostData = (double*)calloc(tab->rows * tab->cols, sizeof(double));
    tab->hostBasicVars = (int*)malloc(lp->numConstraints * sizeof(int));
    
    // Fill tableau
    int slackIdx = lp->numVars;
    int artificialIdx = lp->numVars + numSlack + numSurplus;
    
    for (int i = 0; i < lp->numConstraints; i++) {
        int row = i + 1;  // Skip objective row
        
        // Original variable coefficients (already flipped if RHS was negative)
        for (int j = 0; j < lp->numVars; j++) {
            tab->hostData[row * tab->cols + j] = lp->constraintMatrix[i][j];
        }
        
        // Add slack/surplus/artificial variables
        switch (lp->constraintTypes[i]) {
            case CONSTRAINT_LE:
                // Add slack variable (basic)
                tab->hostData[row * tab->cols + slackIdx] = 1.0;
                tab->hostBasicVars[i] = slackIdx;
                slackIdx++;
                break;
                
            case CONSTRAINT_GE:
                // Subtract surplus variable
                tab->hostData[row * tab->cols + slackIdx] = -1.0;
                slackIdx++;
                // Add artificial variable (basic)
                tab->hostData[row * tab->cols + artificialIdx] = 1.0;
                tab->hostBasicVars[i] = artificialIdx;
                artificialIdx++;
                break;
                
            case CONSTRAINT_EQ:
                // Add artificial variable (basic)
                tab->hostData[row * tab->cols + artificialIdx] = 1.0;
                tab->hostBasicVars[i] = artificialIdx;
                artificialIdx++;
                break;
        }
        
        // RHS (last column) - already non-negative after pre-pass
        tab->hostData[row * tab->cols + (tab->cols - 1)] = lp->rhs[i];
    }
    
    // Symbolic perturbation: add tiny epsilon to RHS to break degeneracy
    // This prevents cycling and reduces numerical errors from degenerate pivots
    for (int i = 0; i < lp->numConstraints; i++) {
        int row = i + 1;
        tab->hostData[row * tab->cols + (tab->cols - 1)] += (i + 1) * PERTURB_EPS;
    }
    
    // Set objective row (will be overwritten for Phase 1)
    // For maximization, negate coefficients (we always minimize internally)
    double objSign = (lp->sense == MAXIMIZE) ? -1.0 : 1.0;
    for (int j = 0; j < lp->numVars; j++) {
        tab->hostData[j] = objSign * lp->objCoeffs[j];
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&tab->data, tableauSize));
    CUDA_CHECK(cudaMalloc(&tab->basicVars, lp->numConstraints * sizeof(int)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData, tableauSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tab->basicVars, tab->hostBasicVars, 
                          lp->numConstraints * sizeof(int), cudaMemcpyHostToDevice));
    
    return tab;
}

void freeTableau(Tableau* tab) {
    if (!tab) return;
    
    CUDA_CHECK(cudaFree(tab->data));
    CUDA_CHECK(cudaFree(tab->basicVars));
    free(tab->hostData);
    free(tab->hostBasicVars);
    free(tab);
}

void syncTableauToHost(Tableau* tab) {
    size_t tableauSize = tab->rows * tab->cols * sizeof(double);
    CUDA_CHECK(cudaMemcpy(tab->hostData, tab->data, tableauSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tab->hostBasicVars, tab->basicVars, 
                          (tab->rows - 1) * sizeof(int), cudaMemcpyDeviceToHost));
}

/**
 * Get a human-readable label for a variable index.
 * Original vars: x0, x1, ...; Slack/surplus: s0, s1, ...; Artificial: a0, a1, ...
 */
static void getVarLabel(Tableau* tab, int varIdx, char* buf, int bufSize) {
    if (varIdx < tab->numOriginalVars) {
        snprintf(buf, bufSize, "x%d", varIdx);
    } else if (varIdx < tab->numOriginalVars + tab->numSlack) {
        snprintf(buf, bufSize, "s%d", varIdx - tab->numOriginalVars);
    } else {
        snprintf(buf, bufSize, "a%d", varIdx - tab->numOriginalVars - tab->numSlack);
    }
}

void printTableau(Tableau* tab) {
    syncTableauToHost(tab);
    
    int rhsCol = tab->cols - 1;
    char label[16];
    
    // --- Column headers ---
    printf("\n%10s|", "Basis");
    for (int j = 0; j < tab->cols; j++) {
        if (j == rhsCol)
            printf("%10s", "RHS");
        else {
            getVarLabel(tab, j, label, sizeof(label));
            printf("%10s", label);
        }
    }
    printf("\n");
    
    // --- Header separator ---
    printf("----------+");
    for (int j = 0; j < tab->cols; j++) printf("----------");
    printf("\n");
    
    // --- Objective row (row 0) ---
    printf("%10s|", "z");
    for (int j = 0; j < tab->cols; j++) {
        printf("%10.3f", tab->hostData[j]);
    }
    printf("\n");
    
    // --- Separator between objective and constraint rows ---
    printf("----------+");
    for (int j = 0; j < tab->cols; j++) printf("----------");
    printf("\n");
    
    // --- Constraint rows ---
    for (int i = 1; i < tab->rows; i++) {
        getVarLabel(tab, tab->hostBasicVars[i - 1], label, sizeof(label));
        printf("%10s|", label);
        for (int j = 0; j < tab->cols; j++) {
            printf("%10.3f", tab->hostData[i * tab->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Print the tableau with iteration context.
 * iteration == 0 means "initial tableau"; pivotRow/pivotCol < 0 means no pivot info.
 */
void printTableauStep(Tableau* tab, int iteration, int pivotRow, int pivotCol) {
    if (iteration == 0) {
        printf("\n>>> Initial Tableau\n");
    } else {
        char enterLabel[16], leaveLabel[16];
        if (pivotCol >= 0) getVarLabel(tab, pivotCol, enterLabel, sizeof(enterLabel));
        else strcpy(enterLabel, "?");
        if (pivotRow > 0) {
            syncTableauToHost(tab);
            getVarLabel(tab, tab->hostBasicVars[pivotRow - 1], leaveLabel, sizeof(leaveLabel));
        } else {
            strcpy(leaveLabel, "?");
        }
        printf("\n>>> Iteration %d  |  Entering: %s (col %d)  Leaving: %s (row %d)\n",
               iteration, enterLabel, pivotCol, leaveLabel, pivotRow);
    }
    printTableau(tab);
}

// ===========================================================================
// TWO-PHASE SIMPLEX ALGORITHM
// ===========================================================================

/**
 * Setup Phase 1: Minimize sum of artificial variables
 */
void setupPhase1(Tableau* tab, double* originalObjective) {
    syncTableauToHost(tab);
    
    // Save original objective
    memcpy(originalObjective, tab->hostData, tab->cols * sizeof(double));
    
    // Set Phase 1 objective: minimize sum of artificial variables
    // This means coefficients of artificial vars are 1, others are 0
    memset(tab->hostData, 0, tab->cols * sizeof(double));
    
    int artificialStart = tab->numOriginalVars + tab->numSlack;
    for (int j = artificialStart; j < artificialStart + tab->numArtificial; j++) {
        tab->hostData[j] = 1.0;
    }
    
    // Make objective row canonical (eliminate artificial basic variables)
    // For each artificial variable in the basis, subtract its row from objective
    for (int i = 0; i < tab->rows - 1; i++) {
        int basicVar = tab->hostBasicVars[i];
        if (basicVar >= artificialStart && basicVar < artificialStart + tab->numArtificial) {
            // Subtract row (i+1) from objective row
            for (int j = 0; j < tab->cols; j++) {
                tab->hostData[j] -= tab->hostData[(i + 1) * tab->cols + j];
            }
        }
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData, 
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
}

/**
 * Setup Phase 2: Restore original objective
 */
void setupPhase2(Tableau* tab, const double* originalObjective) {
    syncTableauToHost(tab);
    
    // Restore original objective
    memcpy(tab->hostData, originalObjective, tab->cols * sizeof(double));
    
    // Make objective row canonical (eliminate current basic variables from objective)
    for (int i = 0; i < tab->rows - 1; i++) {
        int basicVar = tab->hostBasicVars[i];
        double objCoeff = tab->hostData[basicVar];
        
        if (fabs(objCoeff) > EPSILON) {
            // Subtract scaled row from objective
            for (int j = 0; j < tab->cols; j++) {
                tab->hostData[j] -= objCoeff * tab->hostData[(i + 1) * tab->cols + j];
            }
        }
    }
    
    // Set artificial variable coefficients to large positive value (to prevent re-entry)
    int artificialStart = tab->numOriginalVars + tab->numSlack;
    for (int j = artificialStart; j < artificialStart + tab->numArtificial; j++) {
        // Only block non-basic artificials
        int isBasic = 0;
        for (int i = 0; i < tab->rows - 1; i++) {
            if (tab->hostBasicVars[i] == j) { isBasic = 1; break; }
        }
        if (!isBasic) {
            tab->hostData[j] = BIG_M;  // Block from entering
        }
        // Basic artificials should have reduced cost 0 (already handled by canonicalization)
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData, 
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
}

static int pivotOnHost(Tableau* tab, int pivotRow, int pivotCol) {
    double pivot = tab->hostData[pivotRow * tab->cols + pivotCol];
    if (fabs(pivot) <= EPSILON) return 0;

    for (int j = 0; j < tab->cols; j++) {
        tab->hostData[pivotRow * tab->cols + j] /= pivot;
    }

    for (int i = 0; i < tab->rows; i++) {
        if (i == pivotRow) continue;
        double factor = tab->hostData[i * tab->cols + pivotCol];
        if (fabs(factor) <= EPSILON) continue;
        for (int j = 0; j < tab->cols; j++) {
            tab->hostData[i * tab->cols + j] -= factor * tab->hostData[pivotRow * tab->cols + j];
        }
    }

    return 1;
}

static int eliminateDegenerateArtificialRowCol(Tableau* tab, int constraintRowIdx, int artCol) {
    int oldRows = tab->rows;
    int oldCols = tab->cols;
    int removeRow = constraintRowIdx + 1;
    int newRows = oldRows - 1;
    int newCols = oldCols - 1;

    if (newRows < 2 || newCols < 2) return 0;

    double* newHostData = (double*)calloc(newRows * newCols, sizeof(double));
    int* newHostBasicVars = (int*)malloc((newRows - 1) * sizeof(int));
    if (!newHostData || !newHostBasicVars) {
        free(newHostData);
        free(newHostBasicVars);
        return 0;
    }

    int nr = 0;
    for (int r = 0; r < oldRows; r++) {
        if (r == removeRow) continue;
        int nc = 0;
        for (int c = 0; c < oldCols; c++) {
            if (c == artCol) continue;
            newHostData[nr * newCols + nc] = tab->hostData[r * oldCols + c];
            nc++;
        }
        nr++;
    }

    int nb = 0;
    for (int i = 0; i < oldRows - 1; i++) {
        if (i == constraintRowIdx) continue;
        int bv = tab->hostBasicVars[i];
        if (bv == artCol) {
            free(newHostData);
            free(newHostBasicVars);
            return 0;
        }
        if (bv > artCol) bv--;
        newHostBasicVars[nb++] = bv;
    }

    free(tab->hostData);
    free(tab->hostBasicVars);
    CUDA_CHECK(cudaFree(tab->data));
    CUDA_CHECK(cudaFree(tab->basicVars));

    tab->hostData = newHostData;
    tab->hostBasicVars = newHostBasicVars;
    tab->rows = newRows;
    tab->cols = newCols;
    tab->numArtificial--;

    CUDA_CHECK(cudaMalloc(&tab->data, tab->rows * tab->cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tab->basicVars, (tab->rows - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tab->basicVars, tab->hostBasicVars,
                          (tab->rows - 1) * sizeof(int), cudaMemcpyHostToDevice));

    return 1;
}

// Return bitmask: 1 = changed by pivot, 2 = reduced tableau size by row/col elimination
static int extractArtificialBasis(Tableau* tab, SolverConfig* config) {
    syncTableauToHost(tab);

    int status = 0;
    while (1) {
        int changedThisPass = 0;
        int artificialStart = tab->numOriginalVars + tab->numSlack;
        int artificialEnd = artificialStart + tab->numArtificial;
        int rhsCol = tab->cols - 1;

        for (int i = 0; i < tab->rows - 1; i++) {
            int basicVar = tab->hostBasicVars[i];
            if (basicVar < artificialStart || basicVar >= artificialEnd) continue;

            int row = i + 1;
            int entering = -1;
            for (int j = 0; j < artificialStart; j++) {
                if (fabs(tab->hostData[row * tab->cols + j]) > EPSILON) {
                    entering = j;
                    break;
                }
            }

            if (entering >= 0) {
                if (!pivotOnHost(tab, row, entering)) return -1;
                tab->hostBasicVars[i] = entering;
                changedThisPass = 1;
                status |= 1;
                if (config->verbose >= 2) {
                    printf("[DIAG] Extracted artificial var via pivot: row %d, enter col %d\n", row, entering);
                }
                break;
            }

            double rhsVal = tab->hostData[row * tab->cols + rhsCol];
            if (fabs(rhsVal) > 1e-8) {
                if (config->verbose) {
                    printf("Error: Artificial variable in basis has non-zero value (row %d, rhs %.6e)\n",
                           row, rhsVal);
                }
                return -1;
            }

            int safeToDrop = 1;
            for (int c = 0; c < tab->cols - 1; c++) {
                if (c == basicVar) continue;
                if (fabs(tab->hostData[row * tab->cols + c]) > 1e-8) {
                    safeToDrop = 0;
                    break;
                }
            }

            if (!safeToDrop) {
                if (config->verbose >= 2) {
                    printf("[DIAG] Kept degenerate artificial in basis at row %d (row not removable safely)\n", row);
                }
                continue;
            }

            if (!eliminateDegenerateArtificialRowCol(tab, i, basicVar)) return -1;
            changedThisPass = 1;
            status |= 2;
            if (config->verbose >= 2) {
                printf("[DIAG] Removed degenerate row %d and artificial column %d\n", row, basicVar);
            }
            break;
        }

        if (!changedThisPass) break;
    }

    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tab->basicVars, tab->hostBasicVars,
                          (tab->rows - 1) * sizeof(int), cudaMemcpyHostToDevice));

    return status;
}

/**
 * Re-derive the objective row from scratch using the phase cost vector.
 *
 * This combats numerical drift that accumulates over many pivot operations.
 * For each column j:   z_j = phaseCost[j] - sum_i{ phaseCost[basic[i]] * tableau[i+1][j] }
 *
 * @param tab       Tableau (synced to host before call)
 * @param phaseCosts   Cost vector for the current phase (size: tab->cols)
 * @param blockArt  If true, set non-basic artificial columns to +inf after re-derivation
 */
void rederiveObjectiveRow(Tableau* tab, const double* phaseCosts, int blockArt) {
    // Recompute objective row on the host
    for (int j = 0; j < tab->cols; j++) {
        double val = phaseCosts[j];  // original cost for column j
        for (int i = 0; i < tab->rows - 1; i++) {
            int bv = tab->hostBasicVars[i];
            double cb = phaseCosts[bv];  // cost of basic variable
            if (fabs(cb) > EPSILON) {
                val -= cb * tab->hostData[(i + 1) * tab->cols + j];
            }
        }
        tab->hostData[j] = val;
    }
    
    // Block non-basic artificial columns from entering the basis
    if (blockArt) {
        int artStart = tab->numOriginalVars + tab->numSlack;
        for (int j = artStart; j < artStart + tab->numArtificial; j++) {
            int isBasic = 0;
            for (int i = 0; i < tab->rows - 1; i++) {
                if (tab->hostBasicVars[i] == j) { isBasic = 1; break; }
            }
            if (!isBasic) {
                tab->hostData[j] = BIG_M;
            }
        }
    }
    
    // Copy updated objective row back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                          tab->cols * sizeof(double), cudaMemcpyHostToDevice));
}

// ===========================================================================
// PERIODIC CONSTRAINT COLUMN REFACTORIZATION
// ===========================================================================

/**
 * Context for periodic Phase 2 constraint column refactorization.
 * Built once after Phase 1 and passed into runSimplexPhase.
 */
typedef struct {
    LPProblem*    lp;
    const int*    initialBasisCol;  /* col index of initial basis var for constraint k */
    const int*    varToConstraint;  /* for slack/surplus col j: which constraint row it belongs to */
    const double* varCoeffSign;     /* +1 (slack/artificial) or -1 (surplus) */
    int           artStart;         /* first artificial variable column index */
} RefactorContext;

/* Return the original coefficient of variable j in constraint row i from the
 * un-modified LP data.  Works for original vars, slacks, surpluses, and
 * artificials (which all have at most one non-zero entry in their original col). */
static inline double getOrigAij(int j, int i, const LPProblem* lp,
                                 const int* varToConstraint,
                                 const double* varCoeffSign)
{
    if (j < lp->numVars) return lp->constraintMatrix[i][j];
    return (varToConstraint[j] == i) ? varCoeffSign[j] : 0.0;
}

/**
 * Full basis refactorization via Gauss-Jordan elimination.
 *
 * Rebuilds B from original LP data (zero accumulated drift), inverts it
 * exactly, then re-derives every non-basic column and the RHS using the
 * fresh B^{-1}.  Basic-variable columns are restored to canonical identity
 * form.  Artificial variable columns (index >= artStart) are skipped since
 * they are kept blocked by BIG_M in the objective row.
 */
static void refactorConstraintColumns(Tableau* tab, const RefactorContext* rfc, int verbose) {
    LPProblem*    lp              = rfc->lp;
    const int*    varToConstraint = rfc->varToConstraint;
    const double* varCoeffSign    = rfc->varCoeffSign;
    int artStart = rfc->artStart;
    int nCols = tab->cols;
    int rhsCol = nCols - 1;

    /* ---- Build augmented matrix [B | I] and reduce to [I | B^{-1}] ---- */
    /* Row-major; B occupies columns 0..lp->numConstraints-1 and I occupies lp->numConstraints..2*lp->numConstraints-1 */
    double* work = (double*)calloc(lp->numConstraints * 2 * lp->numConstraints, sizeof(double));

    for (int r = 0; r < lp->numConstraints; r++) {
        int bv = tab->hostBasicVars[r];
        /* Column r of B = original column of the r-th basic variable */
        for (int i = 0; i < lp->numConstraints; i++)
            work[i * 2*lp->numConstraints + r] = getOrigAij(bv, i, lp, varToConstraint, varCoeffSign);
        /* Identity block */
        work[r * 2*lp->numConstraints + lp->numConstraints + r] = 1.0;
    }

    /* Gauss-Jordan with partial (column) pivoting */
    for (int col = 0; col < lp->numConstraints; col++) {
        /* Find the row with the largest absolute value in column col */
        int prow = col;
        double best = fabs(work[col * 2*lp->numConstraints + col]);
        for (int row = col + 1; row < lp->numConstraints; row++) {
            double v = fabs(work[row * 2*lp->numConstraints + col]);
            if (v > best) { best = v; prow = row; }
        }
        if (best < 1e-12) {
            /* Near-singular basis — skip this refactorization cycle */
            if (verbose >= 2)
                printf("[DIAG] Refactorization skipped: near-singular basis (col %d, pivot=%.2e)\n",
                       col, best);
            free(work);
            return;
        }
        /* Swap rows prow <-> col */
        if (prow != col) {
            for (int j = 0; j < 2*lp->numConstraints; j++) {
                double tmp = work[col  * 2*lp->numConstraints + j];
                work[col  * 2*lp->numConstraints + j] = work[prow * 2*lp->numConstraints + j];
                work[prow * 2*lp->numConstraints + j] = tmp;
            }
        }
        /* Normalise pivot row */
        double piv = work[col * 2*lp->numConstraints + col];
        for (int j = 0; j < 2*lp->numConstraints; j++) work[col * 2*lp->numConstraints + j] /= piv;
        /* Eliminate column col from every other row */
        for (int row = 0; row < lp->numConstraints; row++) {
            if (row == col) continue;
            double f = work[row * 2*lp->numConstraints + col];
            if (fabs(f) < 1e-16) continue;
            for (int j = 0; j < 2*lp->numConstraints; j++)
                work[row * 2*lp->numConstraints + j] -= f * work[col * 2*lp->numConstraints + j];
        }
    }
    /* B^{-1} now in work[r * 2*lp->numConstraints + lp->numConstraints + k] for row r, column k */

    /* ---- Restore tableau from fresh B^{-1} ---- */
    int* isBasicVar = (int*)calloc(nCols, sizeof(int));
    for (int r = 0; r < lp->numConstraints; r++) isBasicVar[tab->hostBasicVars[r]] = 1;

    /* Restore basic-variable columns to canonical identity */
    for (int r = 0; r < lp->numConstraints; r++) {
        int bv = tab->hostBasicVars[r];
        for (int r2 = 0; r2 < lp->numConstraints; r2++)
            tab->hostData[(r2 + 1) * nCols + bv] = (r2 == r) ? 1.0 : 0.0;
    }

    /* Re-derive non-basic original and slack/surplus columns (skip artificials) */
    for (int j = 0; j < artStart; j++) {
        if (isBasicVar[j]) continue;
        for (int r = 0; r < lp->numConstraints; r++) {
            double val = 0.0;
            for (int k = 0; k < lp->numConstraints; k++)
                val += work[r * 2*lp->numConstraints + lp->numConstraints + k] *
                       getOrigAij(j, k, lp, varToConstraint, varCoeffSign);
            tab->hostData[(r + 1) * nCols + j] = val;
        }
    }

    /* Re-derive RHS as B^{-1} * b_original and restore basic-variable
     * column identity — but DO NOT add any perturbation here.  The existing
     * RHS perturbation state from Phase 2 setup carries through pivot
     * operations naturally and must not be reset, otherwise the anti-cycling
     * guarantee is broken. */
    for (int r = 0; r < lp->numConstraints; r++) {
        double val = 0.0;
        for (int k = 0; k < lp->numConstraints; k++)
            val += work[r * 2*lp->numConstraints + lp->numConstraints + k] * lp->rhs[k];
        /* Keep the accumulated RHS from the current tableau; only use the
         * fresh value to clamp obviously-wrong large negatives caused by
         * numerical blowup (threshold: 10x the original b range). */
        double cur = tab->hostData[(r + 1) * nCols + rhsCol];
        double freshVal = (val < 0.0) ? 0.0 : val;
        /* If accumulated value is reasonably close to fresh, keep it.
         * If it has drifted badly (wrong sign or extreme magnitude), correct. */
        double bMax = 0.0;
        for (int k = 0; k < lp->numConstraints; k++) if (lp->rhs[k] > bMax) bMax = lp->rhs[k];
        if (cur < -1.0 || cur > freshVal + bMax * 10.0 + 1.0)
            tab->hostData[(r + 1) * nCols + rhsCol] = freshVal;
    }

    free(isBasicVar);
    free(work);

    if (verbose >= 2) printf("[DIAG] Full basis refactorization completed (Gauss-Jordan)\n");

    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                          tab->rows * nCols * sizeof(double), cudaMemcpyHostToDevice));
}

/**
 * Run one phase of the simplex algorithm
 * Returns: OPTIMAL, UNBOUNDED, or ERROR
 *
 * @param phaseCosts  Cost vector for the current phase (size: tab->cols).
 *                    Used for periodic objective row re-derivation.
 *                    Pass NULL to disable re-derivation (not recommended).
 * @param rfc         If non-NULL, enables periodic constraint column refactorization
 *                    every REFACTOR_COL_INTERVAL iterations (Phase 2 only).
 */
SimplexStatus runSimplexPhase(Tableau* tab, int maxIterations, const double* phaseCosts, int blockArt, SolverConfig* config, RunContext* run, const RefactorContext* rfc) {
    // Allocate device memory for kernel outputs
    double *d_minVal, *d_minRatio;
    int *d_pivotCol, *d_pivotRow;
    double *d_pivotRowCache, *d_pivotColCache;
    
    CUDA_CHECK(cudaMalloc(&d_minVal, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minRatio, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pivotCol, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivotRow, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivotRowCache, tab->cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pivotColCache, tab->rows * sizeof(double)));
    
    int h_pivotCol, h_pivotRow;
    double h_minVal, h_minRatio;
    
    SimplexStatus status = OPTIMAL;
    int iteration = 0;
    
    // Configure kernel launch parameters
    dim3 blockDim2D(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2D((tab->cols + TILE_SIZE - 1) / TILE_SIZE, 
                   (tab->rows + TILE_SIZE - 1) / TILE_SIZE);
    
    int maxDim = (tab->rows > tab->cols) ? tab->rows : tab->cols;
    int cacheBlocks = (maxDim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Print the initial tableau before any pivoting
    if (config->debug) printTableauStep(tab, 0, -1, -1);
    
    while (iteration < maxIterations) {
        iteration++;
        
        // Check timeout
        if (config->timeout > 0.0) {
            double now = hpc_gettime();
            if (now - run->solveStartTime >= config->timeout) {
                if (config->verbose)
                    printf("Timeout after %.2f seconds at iteration %d\n",
                           now - run->solveStartTime, iteration);
                status = TIMEOUT;
                break;
            }
        }
        
        // Step 1: Find pivot column (entering variable)
        h_pivotCol = -1;
        CUDA_CHECK(cudaMemcpy(d_pivotCol, &h_pivotCol, sizeof(int), cudaMemcpyHostToDevice));
        
        kernelFindPivotColumnSimple<<<1, BLOCK_SIZE>>>(
            tab->data, tab->cols, d_minVal, d_pivotCol
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_pivotCol, d_pivotCol, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_minVal, d_minVal, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Check for optimality
        if (h_pivotCol < 0) {
            if (config->verbose == 1)
                printf("Iteration %d: Optimal solution found (min reduced cost: %.6f)\n", 
                       iteration, h_minVal);
            if (config->verbose >= 2)
                printf("[DIAG] Phase converged after %d iterations (min reduced cost: %.10e)\n",
                       iteration, h_minVal);
            status = OPTIMAL;
            break;
        }
        
        // Step 2: Find pivot row (leaving variable)
        h_pivotRow = -1;
        CUDA_CHECK(cudaMemcpy(d_pivotRow, &h_pivotRow, sizeof(int), cudaMemcpyHostToDevice));
        
        kernelFindPivotRow<<<1, BLOCK_SIZE>>>(
            tab->data, h_pivotCol, tab->rows, tab->cols,
            d_minRatio, d_pivotRow
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_pivotRow, d_pivotRow, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_minRatio, d_minRatio, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Check for unboundedness
        if (h_pivotRow < 0) {
            // Before declaring unbounded, try recovery:
            // 1. Re-derive objective row, check for negative RHS, skip column if needed
            int recovered = 0;
            
            if (phaseCosts != NULL) {
                syncTableauToHost(tab);
                
                // Check if there are positive entries with negative RHS (numerical issue).
                // Try host-side ratio test with slightly relaxed threshold.
                int bestRow = -1;
                double bestRatio = DBL_MAX;
                double bestPivElem = 0.0;
                double ratioFloor = -HARRIS_TOL;  // Allow very slightly negative ratios only
                for (int r = 1; r < tab->rows; r++) {
                    double aij = tab->hostData[r * tab->cols + h_pivotCol];
                    double bi  = tab->hostData[r * tab->cols + (tab->cols - 1)];
                    if (aij > PIVOT_TOL) {
                        double ratio = bi / aij;
                        if (ratio >= ratioFloor) {
                            if (ratio < bestRatio - HARRIS_TOL ||
                                (ratio < bestRatio + HARRIS_TOL && aij > bestPivElem)) {
                                bestRatio = ratio;
                                bestRow = r;
                                bestPivElem = aij;
                            }
                        }
                    }
                }
                
                if (bestRow >= 0) {
                    // Found a valid pivot row with relaxed test — use it
                    h_pivotRow = bestRow;
                    h_minRatio = bestRatio;
                    recovered = 1;
                    if (config->verbose >= 2 && (iteration % 200 == 0))
                        printf("[DIAG] Recovered pivot at iter %d: row %d, ratio %.6e (neg RHS recovery)\n",
                               iteration, h_pivotRow, h_minRatio);
                } else {
                    // No positive entries at all in pivot column — skip this column
                    // Set its reduced cost to +inf to block it, try next column
                    double bigval = BIG_M;
                    tab->hostData[h_pivotCol] = bigval;
                    CUDA_CHECK(cudaMemcpy(&tab->data[h_pivotCol], &bigval,
                                          sizeof(double), cudaMemcpyHostToDevice));
                    // But keep the blocked column at +inf
                    tab->hostData[h_pivotCol] = bigval;
                    CUDA_CHECK(cudaMemcpy(&tab->data[h_pivotCol], &bigval,
                                          sizeof(double), cudaMemcpyHostToDevice));
                    
                    // Try to find next pivot column
                    kernelFindPivotColumnSimple<<<1, BLOCK_SIZE>>>(
                        tab->data, tab->cols, d_minVal, d_pivotCol
                    );
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(&h_pivotCol, d_pivotCol, sizeof(int), cudaMemcpyDeviceToHost));
                    
                    if (h_pivotCol >= 0) {
                        // Try ratio test with new column (on device)
                        kernelFindPivotRow<<<1, BLOCK_SIZE>>>(
                            tab->data, h_pivotCol, tab->rows, tab->cols,
                            d_minRatio, d_pivotRow
                        );
                        CUDA_CHECK(cudaDeviceSynchronize());
                        CUDA_CHECK(cudaMemcpy(&h_pivotRow, d_pivotRow, sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(&h_minRatio, d_minRatio, sizeof(double), cudaMemcpyDeviceToHost));
                        
                        if (h_pivotRow >= 0) {
                            recovered = 1;
                            if (config->verbose >= 2)
                                printf("[DIAG] Column skip at iter %d: new pivot col %d, row %d\n",
                                       iteration, h_pivotCol, h_pivotRow);
                        }
                    }
                    
                    if (!recovered) {
                        // Restore objective row properly before declaring failure
                        syncTableauToHost(tab);
                        rederiveObjectiveRow(tab, phaseCosts, blockArt);
                    }
                }
            }
            
            if (!recovered) {
                if (config->verbose == 1)
                    printf("Iteration %d: Problem is unbounded (no valid pivot row for column %d)\n",
                           iteration, h_pivotCol);
                if (config->verbose >= 2) {
                    printf("[DIAG] UNBOUNDED at iteration %d, pivot col %d, min reduced cost %.10e\n",
                           iteration, h_pivotCol, h_minVal);
                    syncTableauToHost(tab);
                    printf("[DIAG] Pivot column %d values (row: value, rhs):\n", h_pivotCol);
                    int printed = 0;
                    for (int r = 1; r < tab->rows && printed < 20; r++) {
                        double v = tab->hostData[r * tab->cols + h_pivotCol];
                        double rhs = tab->hostData[r * tab->cols + (tab->cols - 1)];
                        if (fabs(v) > EPSILON) {
                            printf("  row %d: col=%.10e rhs=%.10e\n", r, v, rhs);
                            printed++;
                        }
                    }
                    if (printed == 0) printf("  (all zero or near-zero)\n");
                    // Count negative RHS
                    int negCount = 0;
                    for (int r = 1; r < tab->rows; r++) {
                        if (tab->hostData[r * tab->cols + (tab->cols - 1)] < -EPSILON) negCount++;
                    }
                    printf("[DIAG] Rows with negative RHS: %d / %d\n", negCount, tab->rows - 1);
                    printf("[DIAG] Objective row RHS: %.10e\n", tab->hostData[tab->cols - 1]);
                }
                status = UNBOUNDED;
                break;
            }
        }
        
        if (config->verbose == 1)
            printf("Iteration %d: Pivot at row %d, col %d (ratio: %.6f)\n",
                   iteration, h_pivotRow, h_pivotCol, h_minRatio);
        
        // Step 3: Cache pivot row and column
        kernelCachePivotData<<<cacheBlocks, BLOCK_SIZE>>>(
            tab->data, h_pivotRow, h_pivotCol,
            tab->rows, tab->cols,
            d_pivotRowCache, d_pivotColCache
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 4: Update tableau
        kernelUpdateTableau<<<gridDim2D, blockDim2D>>>(
            tab->data, h_pivotRow, h_pivotCol,
            tab->rows, tab->cols,
            d_pivotRowCache, d_pivotColCache
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Update basic variables list
        int constraintIdx = h_pivotRow - 1;  // Row 0 is objective
        CUDA_CHECK(cudaMemcpy(&tab->basicVars[constraintIdx], &h_pivotCol, 
                              sizeof(int), cudaMemcpyHostToDevice));
        tab->hostBasicVars[constraintIdx] = h_pivotCol;
        
        // Track iteration count and log if enabled
        run->totalIterations++;
        if (config->iterLog) {
            double objRHS;
            CUDA_CHECK(cudaMemcpy(&objRHS, &tab->data[tab->cols - 1], sizeof(double), cudaMemcpyDeviceToHost));
            fprintf(config->iterLog, "%d,%d,%d,%d,%.10e,%.10e,%.10e\n",
                    run->totalIterations, run->phase, h_pivotCol, h_pivotRow, h_minVal, h_minRatio, objRHS);
        }
        
        // Periodic objective row re-derivation, and full constraint column
        // refactorization every REFACTOR_COL_INTERVAL iterations if context is available.
        if (phaseCosts != NULL && (iteration % REFACTOR_INTERVAL) == 0) {
            syncTableauToHost(tab);
            if (rfc != NULL && (iteration % REFACTOR_COL_INTERVAL) == 0) {
                refactorConstraintColumns(tab, rfc, config->verbose);
            }
            rederiveObjectiveRow(tab, phaseCosts, blockArt);
            if (config->verbose >= 2 && (iteration % (REFACTOR_INTERVAL * 10)) == 0)
                printf("[DIAG] Re-derived objective row at iteration %d\n", iteration);
        }
        
        // Print tableau after this pivot
        if (config->debug) printTableauStep(tab, iteration, h_pivotRow, h_pivotCol);
    }
    
    // Print final tableau in debug mode
    if (config->debug && iteration > 0) {
        printf("\n>>> Final Tableau (after %d iterations, status: %s)\n",
               iteration, (status == OPTIMAL) ? "OPTIMAL" : "in-progress");
        printTableau(tab);
    }
    
    if (iteration >= maxIterations) {
        printf("Warning: Maximum iterations (%d) reached\n", maxIterations);
        status = ERROR;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_minVal));
    CUDA_CHECK(cudaFree(d_minRatio));
    CUDA_CHECK(cudaFree(d_pivotCol));
    CUDA_CHECK(cudaFree(d_pivotRow));
    CUDA_CHECK(cudaFree(d_pivotRowCache));
    CUDA_CHECK(cudaFree(d_pivotColCache));
    
    return status;
}

/**
 * Main two-phase simplex solver
 */
SimplexStatus solveSimplex(Tableau* tab, LPProblem* lp, SolverConfig* config, RunContext* run) {
    if (config->verbose) printf("\n=== Starting Two-Phase Simplex Method ===\n");
    
    int maxIterations = config->maxIter;
    run->solveStartTime = hpc_gettime();
    
    // Check if Phase 1 is needed
    if (tab->numArtificial > 0) {
        if (config->verbose) printf("\n--- Phase 1: Finding basic feasible solution ---\n");
        
        // Save original objective
        double* originalObjective = (double*)malloc(tab->cols * sizeof(double));
        
        setupPhase1(tab, originalObjective);
        
        // Build Phase 1 cost vector for periodic re-derivation
        // Phase 1 objective: minimize sum of artificial variables
        double* phase1Costs = (double*)calloc(tab->cols, sizeof(double));
        int artStart = tab->numOriginalVars + tab->numSlack;
        for (int j = artStart; j < artStart + tab->numArtificial; j++) {
            phase1Costs[j] = 1.0;
        }
        
        // Diagnostic: check Phase 1 objective row after setup
        if (config->verbose >= 2) {
            syncTableauToHost(tab);
            double initPhase1Obj = tab->hostData[tab->cols - 1];
            printf("[DIAG] Phase 1 initial objective (RHS): %.10f\n", initPhase1Obj);
            int artInBasis = 0;
            for (int i = 0; i < tab->rows - 1; i++) {
                if (tab->hostBasicVars[i] >= artStart) artInBasis++;
            }
            printf("[DIAG] Artificial vars in basis: %d / %d\n", artInBasis, tab->numArtificial);
            // Check for negative RHS values
            int negRHS = 0;
            for (int i = 1; i < tab->rows; i++) {
                double rhs = tab->hostData[i * tab->cols + (tab->cols - 1)];
                if (rhs < -EPSILON) { negRHS++; if (negRHS <= 5) printf("[DIAG] Negative RHS in row %d: %.6f\n", i, rhs); }
            }
            printf("[DIAG] Total rows with negative RHS: %d\n", negRHS);
        }
        
        run->phase = 1;
        SimplexStatus phase1Status = runSimplexPhase(tab, maxIterations, phase1Costs, 0, config, run, NULL);
        free(phase1Costs);
        
        if (phase1Status == UNBOUNDED) {
            printf("Error: Phase 1 should not be unbounded!\n");
            if (config->verbose >= 2) {
                syncTableauToHost(tab);
                printf("[DIAG] Phase 1 ended UNBOUNDED. Obj row RHS: %.10f\n", tab->hostData[tab->cols - 1]);
            }
            free(originalObjective);
            return ERROR;
        }
        
        if (phase1Status != OPTIMAL) {
            free(originalObjective);
            return phase1Status;
        }
        
        // Check if Phase 1 objective is zero (within perturbation tolerance)
        syncTableauToHost(tab);
        double phase1Obj = tab->hostData[tab->cols - 1];  // Objective row, RHS column
        
        // Account for symbolic perturbation: tolerance scales with problem size
        double phase1Tol = (double)(tab->rows) * (double)(tab->rows) * PERTURB_EPS + 1e-6;
        
        if (config->verbose) printf("Phase 1 objective value: %.10f (tolerance: %.6e)\n", phase1Obj, phase1Tol);
        
        if (fabs(phase1Obj) > phase1Tol) {
            if (config->verbose) printf("Problem is INFEASIBLE (Phase 1 objective = %.6f)\n", phase1Obj);
            if (config->verbose >= 2) {
                // Show which artificial vars are still in basis with non-zero values
                int artStart = tab->numOriginalVars + tab->numSlack;
                for (int i = 0; i < tab->rows - 1; i++) {
                    int bv = tab->hostBasicVars[i];
                    double val = tab->hostData[(i+1) * tab->cols + (tab->cols - 1)];
                    if (bv >= artStart && fabs(val) > EPSILON) {
                        printf("[DIAG] Artificial var a%d in basis row %d with value %.10f\n",
                               bv - artStart, i+1, val);
                    }
                }
            }
            free(originalObjective);
            return INFEASIBLE;
        }
        
        int extractionStatus = extractArtificialBasis(tab, config);
        if (extractionStatus < 0) {
            if (config->verbose) printf("Error: Failed to extract degenerate artificial basis\n");
            free(originalObjective);
            return ERROR;
        }
        if (extractionStatus != 0 && config->verbose) {
            printf("Artificial basis cleanup completed before Phase 2\n");
        }
        
        if (config->verbose) printf("\n--- Phase 2: Optimizing original objective ---\n");

        // Arrays kept alive across the Phase 2 run for periodic column refactorization.
        // Allocated inside the if-block below; freed after runSimplexPhase returns.
        int*    rfc_initialBasisCol = NULL;
        int*    rfc_varToConstraint = NULL;
        double* rfc_varCoeffSign    = NULL;
        int     rfc_artStart        = 0;

        // Remove symbolic perturbation and optionally re-derive entire tableau before Phase 2
        if ((extractionStatus & 2) == 0) {
            syncTableauToHost(tab);
            int artStart = tab->numOriginalVars + tab->numSlack;
            
            // Build mapping: constraint index → initial basis column
            // B^{-1} is stored in the columns of the initial basis variables,
            // which are slack columns for LE constraints and artificial columns
            // for GE/EQ constraints.
            int* initialBasisCol = (int*)malloc(lp->numConstraints * sizeof(int));
            {
                int sIdx = lp->numVars;
                int aIdx = artStart;
                for (int c = 0; c < lp->numConstraints; c++) {
                    if (lp->constraintTypes[c] == CONSTRAINT_LE) {
                        initialBasisCol[c] = sIdx;
                        sIdx++;
                    } else if (lp->constraintTypes[c] == CONSTRAINT_GE) {
                        sIdx++;  // skip surplus column
                        initialBasisCol[c] = aIdx;
                        aIdx++;
                    } else { // CONSTRAINT_EQ
                        initialBasisCol[c] = aIdx;
                        aIdx++;
                    }
                }
            }
            
            // Precompute variable-to-constraint mapping for slack/surplus/artificial
            int* varToConstraint = (int*)malloc(tab->cols * sizeof(int));
            double* varCoeffSign = (double*)malloc(tab->cols * sizeof(double));
            for (int j = 0; j < tab->cols; j++) { varToConstraint[j] = -1; varCoeffSign[j] = 0.0; }
            {
                int sIdx = lp->numVars;
                for (int c = 0; c < lp->numConstraints; c++) {
                    if (lp->constraintTypes[c] == CONSTRAINT_LE) {
                        varToConstraint[sIdx] = c; varCoeffSign[sIdx] = 1.0; sIdx++;
                    } else if (lp->constraintTypes[c] == CONSTRAINT_GE) {
                        varToConstraint[sIdx] = c; varCoeffSign[sIdx] = -1.0; sIdx++;
                    }
                }
                int aIdx = artStart;
                for (int c = 0; c < lp->numConstraints; c++) {
                    if (lp->constraintTypes[c] == CONSTRAINT_GE || lp->constraintTypes[c] == CONSTRAINT_EQ) {
                        varToConstraint[aIdx] = c; varCoeffSign[aIdx] = 1.0; aIdx++;
                    }
                }
            }
            
            // Step 1: Remove perturbation via B^{-1} * perturbation subtraction
            for (int r = 0; r < lp->numConstraints; r++) {
                double correction = 0.0;
                for (int j = 0; j < lp->numConstraints; j++) {
                    double bij = tab->hostData[(r + 1) * tab->cols + initialBasisCol[j]];
                    correction += bij * (j + 1) * PERTURB_EPS;
                }
                tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] -= correction;
            }
            
            // Step 2: Check B^{-1} accuracy via one round of iterative refinement
            double maxCorrection = 0.0;
            {
                double* residual = (double*)calloc(lp->numConstraints, sizeof(double));
                for (int i = 0; i < lp->numConstraints; i++) {
                    residual[i] = lp->rhs[i];
                    for (int r = 0; r < lp->numConstraints; r++) {
                        int bv = tab->hostBasicVars[r];
                        double xval = tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)];
                        double aij;
                        if (bv < lp->numVars) {
                            aij = lp->constraintMatrix[i][bv];
                        } else {
                            aij = (varToConstraint[bv] == i) ? varCoeffSign[bv] : 0.0;
                        }
                        residual[i] -= aij * xval;
                    }
                }
                
                for (int r = 0; r < lp->numConstraints; r++) {
                    double corr = 0.0;
                    for (int j = 0; j < lp->numConstraints; j++) {
                        double bij = tab->hostData[(r + 1) * tab->cols + initialBasisCol[j]];
                        corr += bij * residual[j];
                    }
                    if (fabs(corr) > maxCorrection) maxCorrection = fabs(corr);
                }
                
                // Only apply corrections if B^{-1} is reasonable
                if (maxCorrection < 1.0) {
                    // Recompute: apply corrections
                    for (int r = 0; r < lp->numConstraints; r++) {
                        double corr = 0.0;
                        for (int j = 0; j < lp->numConstraints; j++) {
                            double bij = tab->hostData[(r + 1) * tab->cols + initialBasisCol[j]];
                            corr += bij * residual[j];
                        }
                        tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] += corr;
                    }
                }
                free(residual);
                
                if (config->verbose >= 2)
                    printf("[DIAG] RHS refinement: max correction = %.6e (%s)\n", 
                           maxCorrection, maxCorrection < 1.0 ? "applied" : "skipped");
            }
            
            // Step 3: If B^{-1} is accurate, re-derive all original-variable columns
            if (maxCorrection < 1.0) {
                // Mark basic variable columns to skip them
                int* isBasicVar = (int*)calloc(tab->cols, sizeof(int));
                for (int r = 0; r < lp->numConstraints; r++) isBasicVar[tab->hostBasicVars[r]] = 1;
                
                for (int j = 0; j < lp->numVars; j++) {
                    if (isBasicVar[j]) continue;
                    for (int r = 0; r < lp->numConstraints; r++) {
                        double val = 0.0;
                        for (int k = 0; k < lp->numConstraints; k++) {
                            double bij = tab->hostData[(r + 1) * tab->cols + initialBasisCol[k]];
                            val += bij * lp->constraintMatrix[k][j];
                        }
                        tab->hostData[(r + 1) * tab->cols + j] = val;
                    }
                }
                
                // Also re-derive slack/surplus columns
                for (int j = lp->numVars; j < artStart; j++) {
                    if (isBasicVar[j]) continue;
                    int cons = varToConstraint[j];
                    double sign = varCoeffSign[j];
                    if (cons >= 0) {
                        for (int r = 0; r < lp->numConstraints; r++) {
                            tab->hostData[(r + 1) * tab->cols + j] = 
                                sign * tab->hostData[(r + 1) * tab->cols + initialBasisCol[cons]];
                        }
                    }
                }
                
                free(isBasicVar);
                if (config->verbose >= 2) printf("[DIAG] Re-derived all constraint columns\n");
            }
            
            // Keep references alive for Phase 2 periodic constraint column refactorization
            rfc_initialBasisCol = initialBasisCol;
            rfc_varToConstraint = varToConstraint;
            rfc_varCoeffSign    = varCoeffSign;
            rfc_artStart        = artStart;

            // Clamp negative RHS to zero (original behaviour, no Phase-2 perturbation).
            for (int r = 0; r < nCons; r++) {
                double* rp = &tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)];
                if (*rp < 0.0) *rp = 0.0;
            }
            
            CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                                  tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
            if (config->verbose >= 2) printf("[DIAG] Phase 2 tableau prepared\n");
        } else if (config->verbose >= 2) {
            printf("[DIAG] Skipped B^-1-based Phase 2 preparation after row/column elimination\n");
        }
        
        double* phase2Objective = (double*)calloc(tab->cols, sizeof(double));
        double objSign = (lp->sense == MAXIMIZE) ? -1.0 : 1.0;
        int objVars = lp->numVars;
        if (objVars > tab->numOriginalVars) objVars = tab->numOriginalVars;
        for (int j = 0; j < objVars; j++) {
            phase2Objective[j] = objSign * lp->objCoeffs[j];
        }

        setupPhase2(tab, phase2Objective);
        
        // Build Phase 2 cost vector for periodic re-derivation
        // Use 0 for artificial variables — blocking is handled by rederiveObjectiveRow's blockArt flag
        double* phase2Costs = (double*)calloc(tab->cols, sizeof(double));
        memcpy(phase2Costs, phase2Objective, tab->cols * sizeof(double));
        // Artificials get cost 0 to avoid Big-M numerical amplification
        // They are blocked from re-entering by rederiveObjectiveRow(blockArt=1)

        free(phase2Objective);
        free(originalObjective);

        // Build RefactorContext for Phase 2 constraint column refactorization
        RefactorContext rfc_data, *rfc_ptr = NULL;
        if (rfc_initialBasisCol != NULL) {
            rfc_data.lp              = lp;
            rfc_data.initialBasisCol = rfc_initialBasisCol;
            rfc_data.varToConstraint = rfc_varToConstraint;
            rfc_data.varCoeffSign    = rfc_varCoeffSign;
            rfc_data.artStart        = rfc_artStart;
            rfc_ptr = &rfc_data;
        }

        // Phase 2
        run->phase = 2;
        SimplexStatus status = runSimplexPhase(tab, maxIterations, phase2Costs, 1, config, run, rfc_ptr);
        free(phase2Costs);
        free(rfc_initialBasisCol);
        free(rfc_varToConstraint);
        free(rfc_varCoeffSign);
        return status;
    } else {
        if (config->verbose) printf("\nNo artificial variables needed - direct optimization\n");
    }
    
    // Single phase (no artificials) — build cost vector from objective row
    double* phaseCosts = (double*)calloc(tab->cols, sizeof(double));
    syncTableauToHost(tab);
    // The original objective coefficients are already in the objective row
    // (before any pivoting, so use what's stored — but we need the un-reduced costs)
    double objSign = (lp->sense == MAXIMIZE) ? -1.0 : 1.0;
    for (int j = 0; j < lp->numVars; j++) {
        phaseCosts[j] = objSign * lp->objCoeffs[j];
    }
    
    run->phase = 0;
    SimplexStatus status = runSimplexPhase(tab, maxIterations, phaseCosts, 0, config, run, NULL);
    free(phaseCosts);
    return status;
}

/**
 * Extract and print the solution
 */
void printSolution(Tableau* tab, LPProblem* lp, SimplexStatus status) {
    printf("\n=== Solution ===\n");
    
    switch (status) {
        case OPTIMAL:
            printf("Status: OPTIMAL\n\n");
            break;
        case INFEASIBLE:
            printf("Status: INFEASIBLE\n");
            return;
        case UNBOUNDED:
            printf("Status: UNBOUNDED\n");
            return;
        case TIMEOUT:
            printf("Status: TIMEOUT\n");
            return;
        case ERROR:
            printf("Status: ERROR\n");
            return;
    }
    
    syncTableauToHost(tab);
    
    // Extract solution
    double* solution = (double*)calloc(tab->numOriginalVars, sizeof(double));
    
    for (int i = 0; i < tab->rows - 1; i++) {
        int basicVar = tab->hostBasicVars[i];
        if (basicVar < tab->numOriginalVars) {
            solution[basicVar] = tab->hostData[(i + 1) * tab->cols + (tab->cols - 1)];
        }
    }
    
    // Print variable values
    printf("Variable Values:\n");
    for (int i = 0; i < lp->numVars; i++) {
        if (fabs(solution[i]) > EPSILON) {
            printf("  %s = %.6f\n", lp->varNames[i], solution[i]);
        }
    }
    
    // Calculate and print objective value
    double objValue = lp->objConstant;
    for (int i = 0; i < lp->numVars; i++) {
        objValue += lp->objCoeffs[i] * solution[i];
    }
    
    printf("\nObjective Value: %.6f\n", objValue);
    if (lp->objConstant != 0.0)
        printf("  (includes constant term: %.6f)\n", lp->objConstant);
    
    free(solution);
}

