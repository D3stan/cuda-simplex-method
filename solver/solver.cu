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
Tableau* createTableau(LPProblem* lp) {
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
    
    if (g_verbose) {
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
            tab->hostData[j] = 1e20;  // Block from entering
        }
        // Basic artificials should have reduced cost 0 (already handled by canonicalization)
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData, 
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
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
                tab->hostData[j] = 1e20;
            }
        }
    }
    
    // Copy updated objective row back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                          tab->cols * sizeof(double), cudaMemcpyHostToDevice));
}

/**
 * Run one phase of the simplex algorithm
 * Returns: OPTIMAL, UNBOUNDED, or ERROR
 *
 * @param phaseCosts  Cost vector for the current phase (size: tab->cols).
 *                    Used for periodic objective row re-derivation.
 *                    Pass NULL to disable re-derivation (not recommended).
 */
SimplexStatus runSimplexPhase(Tableau* tab, int maxIterations, const double* phaseCosts, int blockArt) {
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
    if (g_debug) printTableauStep(tab, 0, -1, -1);
    
    while (iteration < maxIterations) {
        iteration++;
        
        // Check timeout
        if (g_timeout > 0.0) {
            double now = hpc_gettime();
            if (now - g_solveStartTime >= g_timeout) {
                if (g_verbose)
                    printf("Timeout after %.2f seconds at iteration %d\n",
                           now - g_solveStartTime, iteration);
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
            if (g_verbose == 1)
                printf("Iteration %d: Optimal solution found (min reduced cost: %.6f)\n", 
                       iteration, h_minVal);
            if (g_verbose >= 2)
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
                    if (g_verbose >= 2 && (iteration % 200 == 0))
                        printf("[DIAG] Recovered pivot at iter %d: row %d, ratio %.6e (neg RHS recovery)\n",
                               iteration, h_pivotRow, h_minRatio);
                } else {
                    // No positive entries at all in pivot column — skip this column
                    // Set its reduced cost to +inf to block it, try next column
                    double bigval = 1e20;
                    tab->hostData[h_pivotCol] = bigval;
                    CUDA_CHECK(cudaMemcpy(&tab->data[h_pivotCol], &bigval,
                                          sizeof(double), cudaMemcpyHostToDevice));
                    
                    // Re-derive objective to get correct costs for other columns
                    rederiveObjectiveRow(tab, phaseCosts, blockArt);
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
                            if (g_verbose >= 2)
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
                if (g_verbose == 1)
                    printf("Iteration %d: Problem is unbounded (no valid pivot row for column %d)\n",
                           iteration, h_pivotCol);
                if (g_verbose >= 2) {
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
        
        if (g_verbose == 1)
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
        g_totalIterations++;
        if (g_iterLog) {
            double objRHS;
            CUDA_CHECK(cudaMemcpy(&objRHS, &tab->data[tab->cols - 1], sizeof(double), cudaMemcpyDeviceToHost));
            fprintf(g_iterLog, "%d,%d,%d,%d,%.10e,%.10e,%.10e\n",
                    g_totalIterations, g_phase, h_pivotCol, h_pivotRow, h_minVal, h_minRatio, objRHS);
        }
        
        // Periodic objective row re-derivation to combat numerical drift
        if (phaseCosts != NULL && (iteration % REFACTOR_INTERVAL) == 0) {
            syncTableauToHost(tab);
            rederiveObjectiveRow(tab, phaseCosts, blockArt);
            if (g_verbose >= 2 && (iteration % (REFACTOR_INTERVAL * 10)) == 0)
                printf("[DIAG] Re-derived objective row at iteration %d\n", iteration);
        }
        
        // Print tableau after this pivot
        if (g_debug) printTableauStep(tab, iteration, h_pivotRow, h_pivotCol);
    }
    
    // Print final tableau in debug mode
    if (g_debug && iteration > 0) {
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
SimplexStatus solveSimplex(Tableau* tab, LPProblem* lp) {
    if (g_verbose) printf("\n=== Starting Two-Phase Simplex Method ===\n");
    
    int maxIterations = g_maxIter;
    g_solveStartTime = hpc_gettime();
    
    // Check if Phase 1 is needed
    if (tab->numArtificial > 0) {
        if (g_verbose) printf("\n--- Phase 1: Finding basic feasible solution ---\n");
        
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
        if (g_verbose >= 2) {
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
        
        g_phase = 1;
        SimplexStatus phase1Status = runSimplexPhase(tab, maxIterations, phase1Costs, 0);
        free(phase1Costs);
        
        if (phase1Status == UNBOUNDED) {
            printf("Error: Phase 1 should not be unbounded!\n");
            if (g_verbose >= 2) {
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
        
        if (g_verbose) printf("Phase 1 objective value: %.10f (tolerance: %.6e)\n", phase1Obj, phase1Tol);
        
        if (fabs(phase1Obj) > phase1Tol) {
            if (g_verbose) printf("Problem is INFEASIBLE (Phase 1 objective = %.6f)\n", phase1Obj);
            if (g_verbose >= 2) {
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
        
        // Check if any artificial variable is still in basis
        int artificialStart = tab->numOriginalVars + tab->numSlack;
        for (int i = 0; i < tab->rows - 1; i++) {
            if (tab->hostBasicVars[i] >= artificialStart) {
                if (g_verbose)
                    printf("Warning: Artificial variable %d still in basis (degenerate)\n",
                           tab->hostBasicVars[i]);
            }
        }
        
        if (g_verbose) printf("\n--- Phase 2: Optimizing original objective ---\n");
        
        // Remove symbolic perturbation and optionally re-derive entire tableau before Phase 2
        {
            syncTableauToHost(tab);
            int artStart = tab->numOriginalVars + tab->numSlack;
            int nCons = tab->rows - 1;
            
            // Precompute variable-to-constraint mapping for slack/surplus/artificial
            int* varToConstraint = (int*)malloc(tab->cols * sizeof(int));
            double* varCoeffSign = (double*)malloc(tab->cols * sizeof(double));
            for (int j = 0; j < tab->cols; j++) { varToConstraint[j] = -1; varCoeffSign[j] = 0.0; }
            {
                int sIdx = lp->numVars;
                for (int c = 0; c < nCons; c++) {
                    if (lp->constraintTypes[c] == CONSTRAINT_LE) {
                        varToConstraint[sIdx] = c; varCoeffSign[sIdx] = 1.0; sIdx++;
                    } else if (lp->constraintTypes[c] == CONSTRAINT_GE) {
                        varToConstraint[sIdx] = c; varCoeffSign[sIdx] = -1.0; sIdx++;
                    }
                }
                int aIdx = artStart;
                for (int c = 0; c < nCons; c++) {
                    if (lp->constraintTypes[c] == CONSTRAINT_GE || lp->constraintTypes[c] == CONSTRAINT_EQ) {
                        varToConstraint[aIdx] = c; varCoeffSign[aIdx] = 1.0; aIdx++;
                    }
                }
            }
            
            // Step 1: Remove perturbation via B^{-1} * perturbation subtraction
            for (int r = 0; r < nCons; r++) {
                double correction = 0.0;
                for (int j = 0; j < nCons; j++) {
                    double bij = tab->hostData[(r + 1) * tab->cols + (artStart + j)];
                    correction += bij * (j + 1) * PERTURB_EPS;
                }
                tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] -= correction;
            }
            
            // Step 2: Check B^{-1} accuracy via one round of iterative refinement
            double maxCorrection = 0.0;
            {
                double* residual = (double*)calloc(nCons, sizeof(double));
                for (int i = 0; i < nCons; i++) {
                    residual[i] = lp->rhs[i];
                    for (int r = 0; r < nCons; r++) {
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
                
                for (int r = 0; r < nCons; r++) {
                    double corr = 0.0;
                    for (int j = 0; j < nCons; j++) {
                        double bij = tab->hostData[(r + 1) * tab->cols + (artStart + j)];
                        corr += bij * residual[j];
                    }
                    if (fabs(corr) > maxCorrection) maxCorrection = fabs(corr);
                }
                
                // Only apply corrections if B^{-1} is reasonable
                if (maxCorrection < 1.0) {
                    // Recompute: apply corrections
                    for (int r = 0; r < nCons; r++) {
                        double corr = 0.0;
                        for (int j = 0; j < nCons; j++) {
                            double bij = tab->hostData[(r + 1) * tab->cols + (artStart + j)];
                            corr += bij * residual[j];
                        }
                        tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] += corr;
                    }
                }
                free(residual);
                
                if (g_verbose >= 2)
                    printf("[DIAG] RHS refinement: max correction = %.6e (%s)\n", 
                           maxCorrection, maxCorrection < 1.0 ? "applied" : "skipped");
            }
            
            // Step 3: If B^{-1} is accurate, re-derive all original-variable columns
            if (maxCorrection < 1.0) {
                // Mark basic variable columns to skip them
                int* isBasicVar = (int*)calloc(tab->cols, sizeof(int));
                for (int r = 0; r < nCons; r++) isBasicVar[tab->hostBasicVars[r]] = 1;
                
                for (int j = 0; j < lp->numVars; j++) {
                    if (isBasicVar[j]) continue;
                    for (int r = 0; r < nCons; r++) {
                        double val = 0.0;
                        for (int k = 0; k < nCons; k++) {
                            double bij = tab->hostData[(r + 1) * tab->cols + (artStart + k)];
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
                        for (int r = 0; r < nCons; r++) {
                            tab->hostData[(r + 1) * tab->cols + j] = 
                                sign * tab->hostData[(r + 1) * tab->cols + (artStart + cons)];
                        }
                    }
                }
                
                free(isBasicVar);
                if (g_verbose >= 2) printf("[DIAG] Re-derived all constraint columns\n");
            }
            
            free(varToConstraint);
            free(varCoeffSign);
            
            // Clamp negative RHS to zero
            for (int r = 0; r < nCons; r++) {
                if (tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] < 0.0) {
                    tab->hostData[(r + 1) * tab->cols + (tab->cols - 1)] = 0.0;
                }
            }
            
            CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData,
                                  tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
            if (g_verbose >= 2) printf("[DIAG] Phase 2 tableau prepared\n");
        }
        
        setupPhase2(tab, originalObjective);
        
        // Build Phase 2 cost vector for periodic re-derivation
        // Use 0 for artificial variables — blocking is handled by rederiveObjectiveRow's blockArt flag
        double* phase2Costs = (double*)calloc(tab->cols, sizeof(double));
        memcpy(phase2Costs, originalObjective, tab->cols * sizeof(double));
        // Artificials get cost 0 to avoid Big-M numerical amplification
        // They are blocked from re-entering by rederiveObjectiveRow(blockArt=1)
        
        free(originalObjective);
        
        // Phase 2
        g_phase = 2;
        SimplexStatus status = runSimplexPhase(tab, maxIterations, phase2Costs, 1);
        free(phase2Costs);
        return status;
    } else {
        if (g_verbose) printf("\nNo artificial variables needed - direct optimization\n");
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
    
    g_phase = 0;
    SimplexStatus status = runSimplexPhase(tab, maxIterations, phaseCosts, 0);
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

