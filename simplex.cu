/**
 * CUDA Implementation of the Two-Phase Simplex Method
 * 
 * Features:
 * - Double precision arithmetic
 * - MPS file format input
 * - Handles <=, >=, = constraints
 * - Supports both maximization and minimization
 * 
 * Build: nvcc -o simplex simplex.cu -O3
 * Usage: simplex <problem.mps>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// ===========================================================================
// CONSTANTS AND CONFIGURATION
// ===========================================================================

#define EPSILON 1e-10
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ===========================================================================
// DATA STRUCTURES
// ===========================================================================

// Constraint types
typedef enum {
    CONSTRAINT_LE,  // <=
    CONSTRAINT_GE,  // >=
    CONSTRAINT_EQ   // =
} ConstraintType;

// Optimization sense
typedef enum {
    MINIMIZE,
    MAXIMIZE
} OptSense;

// Result status
typedef enum {
    OPTIMAL,
    INFEASIBLE,
    UNBOUNDED,
    ERROR
} SimplexStatus;

// Linear Programming Problem structure
typedef struct {
    char name[256];
    int numVars;           // Original variables
    int numConstraints;    // Number of constraints
    
    double* objCoeffs;     // Objective coefficients (size: numVars)
    double** constraintMatrix;  // Constraint coefficients (numConstraints x numVars)
    double* rhs;           // Right-hand side values (size: numConstraints)
    ConstraintType* constraintTypes;  // Type of each constraint
    
    OptSense sense;        // MINIMIZE or MAXIMIZE
    
    // Variable names for output
    char** varNames;
    char** constraintNames;
} LPProblem;

// Tableau structure for simplex
typedef struct {
    double* data;          // Flattened tableau (device memory)
    double* hostData;      // Host copy for debugging
    int rows;              // numConstraints + 1 (objective row)
    int cols;              // All variables + RHS column
    int* basicVars;        // Indices of basic variables (device)
    int* hostBasicVars;    // Host copy
    int numOriginalVars;
    int numSlack;
    int numArtificial;
} Tableau;

// ===========================================================================
// MPS PARSER
// ===========================================================================

// Trim whitespace from string
static void trim(char* str) {
    char* start = str;
    while (*start == ' ' || *start == '\t') start++;
    
    char* end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        *end = '\0';
        end--;
    }
    
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

// Parse MPS file
LPProblem* parseMPS(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    LPProblem* lp = (LPProblem*)calloc(1, sizeof(LPProblem));
    lp->sense = MINIMIZE;  // Default
    
    char line[1024];
    char section[64] = "";
    
    // First pass: count rows and columns
    int maxVars = 1000;
    int maxConstraints = 1000;
    
    char** rowNames = (char**)malloc(maxConstraints * sizeof(char*));
    ConstraintType* rowTypes = (ConstraintType*)malloc(maxConstraints * sizeof(ConstraintType));
    char* objRowName = NULL;
    int numRows = 0;
    
    // Temporary storage for columns
    typedef struct {
        char name[64];
        int index;
    } VarEntry;
    VarEntry* varEntries = (VarEntry*)malloc(maxVars * sizeof(VarEntry));
    int numVars = 0;
    
    // Temporary storage for coefficients (sparse)
    typedef struct {
        int row;
        int col;
        double value;
    } CoeffEntry;
    CoeffEntry* coeffs = (CoeffEntry*)malloc(maxVars * maxConstraints * sizeof(CoeffEntry));
    int numCoeffs = 0;
    
    // RHS values
    double* rhsValues = (double*)calloc(maxConstraints, sizeof(double));
    
    // Helper function to find variable index
    auto findVar = [&](const char* name) -> int {
        for (int i = 0; i < numVars; i++) {
            if (strcmp(varEntries[i].name, name) == 0) return i;
        }
        return -1;
    };
    
    // Helper function to find row index
    auto findRow = [&](const char* name) -> int {
        if (objRowName && strcmp(name, objRowName) == 0) return -1; // Objective
        for (int i = 0; i < numRows; i++) {
            if (strcmp(rowNames[i], name) == 0) return i;
        }
        return -2; // Not found
    };
    
    // Parse file
    while (fgets(line, sizeof(line), file)) {
        trim(line);
        if (line[0] == '*' || line[0] == '\0') continue;  // Comment or empty
        
        // Check for section headers
        if (strncmp(line, "NAME", 4) == 0) {
            sscanf(line + 4, "%s", lp->name);
            continue;
        }
        if (strcmp(line, "ROWS") == 0) { strcpy(section, "ROWS"); continue; }
        if (strcmp(line, "COLUMNS") == 0) { strcpy(section, "COLUMNS"); continue; }
        if (strcmp(line, "RHS") == 0) { strcpy(section, "RHS"); continue; }
        if (strcmp(line, "BOUNDS") == 0) { strcpy(section, "BOUNDS"); continue; }
        if (strcmp(line, "RANGES") == 0) { strcpy(section, "RANGES"); continue; }
        if (strcmp(line, "ENDATA") == 0) break;
        if (strncmp(line, "OBJSENSE", 8) == 0) {
            if (fgets(line, sizeof(line), file)) {
                trim(line);
                if (strncmp(line, "MAX", 3) == 0) lp->sense = MAXIMIZE;
            }
            continue;
        }
        
        // Parse based on current section
        if (strcmp(section, "ROWS") == 0) {
            char type;
            char name[64];
            if (sscanf(line, " %c %s", &type, name) == 2) {
                if (type == 'N') {
                    // Objective row
                    objRowName = strdup(name);
                } else {
                    rowNames[numRows] = strdup(name);
                    switch (type) {
                        case 'L': rowTypes[numRows] = CONSTRAINT_LE; break;
                        case 'G': rowTypes[numRows] = CONSTRAINT_GE; break;
                        case 'E': rowTypes[numRows] = CONSTRAINT_EQ; break;
                        default: rowTypes[numRows] = CONSTRAINT_EQ;
                    }
                    numRows++;
                }
            }
        }
        else if (strcmp(section, "COLUMNS") == 0) {
            char colName[64], rowName1[64], rowName2[64];
            double val1, val2;
            
            int parsed = sscanf(line, "%s %s %lf %s %lf", colName, rowName1, &val1, rowName2, &val2);
            
            if (parsed >= 3) {
                // Find or add variable
                int varIdx = findVar(colName);
                if (varIdx < 0) {
                    strcpy(varEntries[numVars].name, colName);
                    varEntries[numVars].index = numVars;
                    varIdx = numVars++;
                }
                
                // Add coefficient
                int rowIdx = findRow(rowName1);
                if (rowIdx == -1) {
                    // Objective coefficient
                    // Will be stored separately
                } else if (rowIdx >= 0) {
                    coeffs[numCoeffs].row = rowIdx;
                    coeffs[numCoeffs].col = varIdx;
                    coeffs[numCoeffs].value = val1;
                    numCoeffs++;
                }
                
                // Second coefficient on same line
                if (parsed == 5) {
                    rowIdx = findRow(rowName2);
                    if (rowIdx == -1) {
                        // Objective
                    } else if (rowIdx >= 0) {
                        coeffs[numCoeffs].row = rowIdx;
                        coeffs[numCoeffs].col = varIdx;
                        coeffs[numCoeffs].value = val2;
                        numCoeffs++;
                    }
                }
            }
        }
        else if (strcmp(section, "RHS") == 0) {
            char rhsName[64], rowName1[64], rowName2[64];
            double val1, val2;
            
            int parsed = sscanf(line, "%s %s %lf %s %lf", rhsName, rowName1, &val1, rowName2, &val2);
            
            if (parsed >= 3) {
                int rowIdx = findRow(rowName1);
                if (rowIdx >= 0) {
                    rhsValues[rowIdx] = val1;
                }
                
                if (parsed == 5) {
                    rowIdx = findRow(rowName2);
                    if (rowIdx >= 0) {
                        rhsValues[rowIdx] = val2;
                    }
                }
            }
        }
    }
    
    fclose(file);
    
    // Now rebuild the file to get objective coefficients
    file = fopen(filename, "r");
    lp->numVars = numVars;
    lp->numConstraints = numRows;
    lp->objCoeffs = (double*)calloc(numVars, sizeof(double));
    lp->rhs = (double*)malloc(numRows * sizeof(double));
    lp->constraintTypes = (ConstraintType*)malloc(numRows * sizeof(ConstraintType));
    lp->constraintMatrix = (double**)malloc(numRows * sizeof(double*));
    lp->varNames = (char**)malloc(numVars * sizeof(char*));
    lp->constraintNames = (char**)malloc(numRows * sizeof(char*));
    
    for (int i = 0; i < numRows; i++) {
        lp->constraintMatrix[i] = (double*)calloc(numVars, sizeof(double));
        lp->rhs[i] = rhsValues[i];
        lp->constraintTypes[i] = rowTypes[i];
        lp->constraintNames[i] = rowNames[i];
    }
    
    for (int i = 0; i < numVars; i++) {
        lp->varNames[i] = strdup(varEntries[i].name);
    }
    
    // Fill constraint matrix from sparse representation
    for (int i = 0; i < numCoeffs; i++) {
        lp->constraintMatrix[coeffs[i].row][coeffs[i].col] = coeffs[i].value;
    }
    
    // Re-read for objective coefficients
    strcpy(section, "");
    while (fgets(line, sizeof(line), file)) {
        trim(line);
        if (line[0] == '*' || line[0] == '\0') continue;
        
        if (strcmp(line, "COLUMNS") == 0) { strcpy(section, "COLUMNS"); continue; }
        if (strcmp(line, "RHS") == 0) break;
        
        if (strcmp(section, "COLUMNS") == 0) {
            char colName[64], rowName1[64], rowName2[64];
            double val1, val2;
            
            int parsed = sscanf(line, "%s %s %lf %s %lf", colName, rowName1, &val1, rowName2, &val2);
            
            if (parsed >= 3) {
                int varIdx = findVar(colName);
                if (objRowName && strcmp(rowName1, objRowName) == 0) {
                    lp->objCoeffs[varIdx] = val1;
                }
                if (parsed == 5 && objRowName && strcmp(rowName2, objRowName) == 0) {
                    lp->objCoeffs[varIdx] = val2;
                }
            }
        }
    }
    
    fclose(file);
    
    // Cleanup temporary storage
    free(rhsValues);
    free(coeffs);
    free(varEntries);
    free(rowTypes);
    free(rowNames);
    if (objRowName) free(objRowName);
    
    printf("Parsed LP: %s\n", lp->name);
    printf("  Variables: %d\n", lp->numVars);
    printf("  Constraints: %d\n", lp->numConstraints);
    printf("  Sense: %s\n", lp->sense == MAXIMIZE ? "MAXIMIZE" : "MINIMIZE");
    
    return lp;
}

void freeLPProblem(LPProblem* lp) {
    if (!lp) return;
    
    free(lp->objCoeffs);
    free(lp->rhs);
    free(lp->constraintTypes);
    
    for (int i = 0; i < lp->numConstraints; i++) {
        free(lp->constraintMatrix[i]);
        free(lp->constraintNames[i]);
    }
    free(lp->constraintMatrix);
    free(lp->constraintNames);
    
    for (int i = 0; i < lp->numVars; i++) {
        free(lp->varNames[i]);
    }
    free(lp->varNames);
    
    free(lp);
}

// ===========================================================================
// CUDA KERNELS
// ===========================================================================

/**
 * Kernel A: Find pivot column (entering variable)
 * Uses parallel reduction to find the most negative reduced cost
 * 
 * @param costs Pointer to the objective row (reduced costs)
 * @param numCols Number of columns to search
 * @param minVal Output: minimum value found
 * @param minIdx Output: index of minimum value (-1 if all non-negative)
 */
__global__ void kernelFindPivotColumn(
    const double* tableau,
    int numCols,
    int rowStride,  // Number of columns in full tableau (for accessing objective row)
    double* minVal,
    int* minIdx
) {
    __shared__ double sharedMin[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with neutral values
    sharedMin[tid] = 0.0;  // We only care about negative values
    sharedIdx[tid] = -1;
    
    // Each thread processes multiple elements if needed
    double localMin = 0.0;
    int localIdx = -1;
    
    for (int i = gid; i < numCols - 1; i += blockDim.x * gridDim.x) {  // -1 to exclude RHS
        double val = tableau[i];  // Objective row is row 0
        if (val < localMin - EPSILON) {
            localMin = val;
            localIdx = i;
        }
    }
    
    sharedMin[tid] = localMin;
    sharedIdx[tid] = localIdx;
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedMin[tid + stride] < sharedMin[tid] - EPSILON) {
                sharedMin[tid] = sharedMin[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicMin(reinterpret_cast<unsigned long long*>(minVal), 
                  __double_as_longlong(sharedMin[0]));
        // Use atomic CAS to update index if this block has the minimum
        if (sharedIdx[0] >= 0) {
            // Simple approach: first block with valid minimum wins
            atomicCAS(minIdx, -1, sharedIdx[0]);
        }
    }
}

// Simpler version of pivot column finding (single block, reliable)
__global__ void kernelFindPivotColumnSimple(
    const double* tableau,
    int numCols,
    double* minVal,
    int* minIdx
) {
    __shared__ double sharedMin[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Initialize
    sharedMin[tid] = 0.0;
    sharedIdx[tid] = -1;
    
    // Each thread checks multiple columns
    for (int col = tid; col < numCols - 1; col += blockDim.x) {
        double val = tableau[col];
        if (val < sharedMin[tid] - EPSILON) {
            sharedMin[tid] = val;
            sharedIdx[tid] = col;
        }
    }
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedMin[tid + stride] < sharedMin[tid] - EPSILON) {
                sharedMin[tid] = sharedMin[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *minVal = sharedMin[0];
        *minIdx = sharedIdx[0];
    }
}

/**
 * Kernel B: Find pivot row (leaving variable)
 * Performs minimum ratio test in parallel
 * 
 * @param tableau Full tableau
 * @param pivotCol Index of pivot column
 * @param numRows Number of constraint rows (excluding objective row)
 * @param numCols Number of columns in tableau
 * @param minRatio Output: minimum ratio found
 * @param pivotRow Output: index of pivot row (-1 if unbounded)
 */
__global__ void kernelFindPivotRow(
    const double* tableau,
    int pivotCol,
    int numRows,      // Total rows including objective
    int numCols,
    double* minRatio,
    int* pivotRow
) {
    __shared__ double sharedRatio[BLOCK_SIZE];
    __shared__ int sharedRow[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Initialize with "infinity"
    sharedRatio[tid] = DBL_MAX;
    sharedRow[tid] = -1;
    
    // Each thread processes one or more rows (skip row 0 = objective)
    for (int row = tid + 1; row < numRows; row += blockDim.x) {
        double aij = tableau[row * numCols + pivotCol];
        double bi = tableau[row * numCols + (numCols - 1)];  // RHS column
        
        // Minimum ratio test: only consider positive coefficients
        if (aij > EPSILON) {
            double ratio = bi / aij;
            // Only consider non-negative ratios (RHS should be non-negative in standard form)
            if (ratio >= -EPSILON && ratio < sharedRatio[tid]) {
                sharedRatio[tid] = ratio;
                sharedRow[tid] = row;
            }
        }
    }
    __syncthreads();
    
    // Reduction to find minimum ratio
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedRatio[tid + stride] < sharedRatio[tid] - EPSILON) {
                sharedRatio[tid] = sharedRatio[tid + stride];
                sharedRow[tid] = sharedRow[tid + stride];
            } else if (fabs(sharedRatio[tid + stride] - sharedRatio[tid]) < EPSILON) {
                // Tie-breaking: prefer smaller row index (Bland's rule for anti-cycling)
                if (sharedRow[tid + stride] < sharedRow[tid] && sharedRow[tid + stride] >= 0) {
                    sharedRow[tid] = sharedRow[tid + stride];
                }
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *minRatio = sharedRatio[0];
        *pivotRow = (sharedRatio[0] < DBL_MAX - 1.0) ? sharedRow[0] : -1;
    }
}

/**
 * Kernel C: Update tableau (pivoting operation)
 * Each thread updates one element of the tableau
 * 
 * Uses the formula:
 *   For pivot row: a'[pivotRow][j] = a[pivotRow][j] / pivotElement
 *   For other rows: a'[i][j] = a[i][j] - a[i][pivotCol] * a'[pivotRow][j]
 */
__global__ void kernelUpdateTableau(
    double* tableau,
    int pivotRow,
    int pivotCol,
    int numRows,
    int numCols,
    const double* pivotRowCache,   // Pre-divided pivot row
    const double* pivotColCache    // Original pivot column values
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= numRows || col >= numCols) return;
    
    if (row == pivotRow) {
        // Pivot row: use pre-computed divided values
        tableau[row * numCols + col] = pivotRowCache[col];
    } else {
        // Other rows: eliminate using pivot row
        double factor = pivotColCache[row];
        tableau[row * numCols + col] -= factor * pivotRowCache[col];
    }
}

/**
 * Kernel to extract pivot row and column for caching
 */
__global__ void kernelCachePivotData(
    const double* tableau,
    int pivotRow,
    int pivotCol,
    int numRows,
    int numCols,
    double* pivotRowCache,
    double* pivotColCache
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    double pivotElement = tableau[pivotRow * numCols + pivotCol];
    
    // Cache pivot row (divided by pivot element)
    if (tid < numCols) {
        pivotRowCache[tid] = tableau[pivotRow * numCols + tid] / pivotElement;
    }
    
    // Cache pivot column (original values)
    if (tid < numRows) {
        pivotColCache[tid] = tableau[tid * numCols + pivotCol];
    }
}

// ===========================================================================
// TABLEAU MANAGEMENT
// ===========================================================================

/**
 * Create and initialize the tableau from LP problem
 * Handles slack, surplus, and artificial variables
 */
Tableau* createTableau(LPProblem* lp) {
    Tableau* tab = (Tableau*)malloc(sizeof(Tableau));
    
    // Count additional variables needed
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
    
    printf("Tableau dimensions: %d x %d\n", tab->rows, tab->cols);
    printf("  Original variables: %d\n", lp->numVars);
    printf("  Slack variables: %d\n", numSlack);
    printf("  Surplus variables: %d\n", numSurplus);
    printf("  Artificial variables: %d\n", numArtificial);
    
    // Allocate host memory
    size_t tableauSize = tab->rows * tab->cols * sizeof(double);
    tab->hostData = (double*)calloc(tab->rows * tab->cols, sizeof(double));
    tab->hostBasicVars = (int*)malloc(lp->numConstraints * sizeof(int));
    
    // Fill tableau
    int slackIdx = lp->numVars;
    int artificialIdx = lp->numVars + numSlack + numSurplus;
    
    for (int i = 0; i < lp->numConstraints; i++) {
        int row = i + 1;  // Skip objective row
        
        // Original variable coefficients
        for (int j = 0; j < lp->numVars; j++) {
            tab->hostData[row * tab->cols + j] = lp->constraintMatrix[i][j];
        }
        
        // Handle RHS sign (we need b >= 0)
        double rhsSign = 1.0;
        if (lp->rhs[i] < 0) {
            rhsSign = -1.0;
            // Multiply entire row by -1
            for (int j = 0; j < lp->numVars; j++) {
                tab->hostData[row * tab->cols + j] *= -1.0;
            }
            // Flip constraint type
            if (lp->constraintTypes[i] == CONSTRAINT_LE) {
                lp->constraintTypes[i] = CONSTRAINT_GE;
            } else if (lp->constraintTypes[i] == CONSTRAINT_GE) {
                lp->constraintTypes[i] = CONSTRAINT_LE;
            }
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
        
        // RHS (last column)
        tab->hostData[row * tab->cols + (tab->cols - 1)] = fabs(lp->rhs[i]);
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

void printTableau(Tableau* tab) {
    syncTableauToHost(tab);
    
    printf("\nTableau (%d x %d):\n", tab->rows, tab->cols);
    for (int i = 0; i < tab->rows; i++) {
        if (i == 0) printf("Obj: ");
        else printf("R%02d: ", i);
        
        for (int j = 0; j < tab->cols; j++) {
            printf("%8.3f ", tab->hostData[i * tab->cols + j]);
        }
        printf("\n");
    }
    
    printf("Basic variables: ");
    for (int i = 0; i < tab->rows - 1; i++) {
        printf("%d ", tab->hostBasicVars[i]);
    }
    printf("\n");
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
        tab->hostData[j] = 1e10;  // Big-M
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(tab->data, tab->hostData, 
                          tab->rows * tab->cols * sizeof(double), cudaMemcpyHostToDevice));
}

/**
 * Run one phase of the simplex algorithm
 * Returns: OPTIMAL, UNBOUNDED, or ERROR
 */
SimplexStatus runSimplexPhase(Tableau* tab, int maxIterations) {
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
    
    while (iteration < maxIterations) {
        iteration++;
        
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
            printf("Iteration %d: Optimal solution found (min reduced cost: %.6f)\n", 
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
            printf("Iteration %d: Problem is unbounded (no valid pivot row for column %d)\n",
                   iteration, h_pivotCol);
            status = UNBOUNDED;
            break;
        }
        
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
    printf("\n=== Starting Two-Phase Simplex Method ===\n");
    
    int maxIterations = 1000;
    
    // Check if Phase 1 is needed
    if (tab->numArtificial > 0) {
        printf("\n--- Phase 1: Finding basic feasible solution ---\n");
        
        // Save original objective
        double* originalObjective = (double*)malloc(tab->cols * sizeof(double));
        
        setupPhase1(tab, originalObjective);
        
        SimplexStatus phase1Status = runSimplexPhase(tab, maxIterations);
        
        if (phase1Status == UNBOUNDED) {
            printf("Error: Phase 1 should not be unbounded!\n");
            free(originalObjective);
            return ERROR;
        }
        
        if (phase1Status != OPTIMAL) {
            free(originalObjective);
            return phase1Status;
        }
        
        // Check if Phase 1 objective is zero
        syncTableauToHost(tab);
        double phase1Obj = tab->hostData[tab->cols - 1];  // Objective row, RHS column
        
        printf("Phase 1 objective value: %.10f\n", phase1Obj);
        
        if (fabs(phase1Obj) > EPSILON) {
            printf("Problem is INFEASIBLE (Phase 1 objective = %.6f)\n", phase1Obj);
            free(originalObjective);
            return INFEASIBLE;
        }
        
        // Check if any artificial variable is still in basis
        int artificialStart = tab->numOriginalVars + tab->numSlack;
        for (int i = 0; i < tab->rows - 1; i++) {
            if (tab->hostBasicVars[i] >= artificialStart) {
                printf("Warning: Artificial variable %d still in basis (degenerate)\n",
                       tab->hostBasicVars[i]);
            }
        }
        
        printf("\n--- Phase 2: Optimizing original objective ---\n");
        setupPhase2(tab, originalObjective);
        
        free(originalObjective);
    } else {
        printf("\nNo artificial variables needed - direct optimization\n");
    }
    
    // Phase 2 (or single phase if no artificials)
    SimplexStatus status = runSimplexPhase(tab, maxIterations);
    
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
    double objValue = 0.0;
    for (int i = 0; i < lp->numVars; i++) {
        objValue += lp->objCoeffs[i] * solution[i];
    }
    
    printf("\nObjective Value: %.6f\n", objValue);
    
    free(solution);
}

// ===========================================================================
// MAIN FUNCTION
// ===========================================================================

/**
 * Create a simple test problem if no file is provided
 * 
 * Maximize: 3x1 + 2x2
 * Subject to:
 *     x1 + x2 <= 4
 *    2x1 + x2 <= 6
 *     x1, x2 >= 0
 * 
 * Optimal: x1 = 2, x2 = 2, z = 10
 */
LPProblem* createTestProblem() {
    LPProblem* lp = (LPProblem*)calloc(1, sizeof(LPProblem));
    
    strcpy(lp->name, "TestProblem");
    lp->numVars = 2;
    lp->numConstraints = 2;
    lp->sense = MAXIMIZE;
    
    lp->objCoeffs = (double*)malloc(2 * sizeof(double));
    lp->objCoeffs[0] = 3.0;
    lp->objCoeffs[1] = 2.0;
    
    lp->constraintMatrix = (double**)malloc(2 * sizeof(double*));
    lp->constraintMatrix[0] = (double*)malloc(2 * sizeof(double));
    lp->constraintMatrix[1] = (double*)malloc(2 * sizeof(double));
    
    lp->constraintMatrix[0][0] = 1.0;
    lp->constraintMatrix[0][1] = 1.0;
    lp->constraintMatrix[1][0] = 2.0;
    lp->constraintMatrix[1][1] = 1.0;
    
    lp->rhs = (double*)malloc(2 * sizeof(double));
    lp->rhs[0] = 4.0;
    lp->rhs[1] = 6.0;
    
    lp->constraintTypes = (ConstraintType*)malloc(2 * sizeof(ConstraintType));
    lp->constraintTypes[0] = CONSTRAINT_LE;
    lp->constraintTypes[1] = CONSTRAINT_LE;
    
    lp->varNames = (char**)malloc(2 * sizeof(char*));
    lp->varNames[0] = strdup("x1");
    lp->varNames[1] = strdup("x2");
    
    lp->constraintNames = (char**)malloc(2 * sizeof(char*));
    lp->constraintNames[0] = strdup("c1");
    lp->constraintNames[1] = strdup("c2");
    
    return lp;
}

int main(int argc, char* argv[]) {
    printf("CUDA Two-Phase Simplex Solver\n");
    printf("=============================\n\n");
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    
    LPProblem* lp = NULL;
    
    if (argc > 1) {
        printf("Loading problem from: %s\n\n", argv[1]);
        lp = parseMPS(argv[1]);
        if (!lp) {
            return EXIT_FAILURE;
        }
    } else {
        printf("No input file provided. Using test problem.\n\n");
        printf("Usage: %s <problem.mps>\n\n", argv[0]);
        lp = createTestProblem();
        
        printf("Test Problem:\n");
        printf("  Maximize: 3*x1 + 2*x2\n");
        printf("  Subject to:\n");
        printf("    x1 + x2 <= 4\n");
        printf("    2*x1 + x2 <= 6\n");
        printf("  Expected: x1=2, x2=2, z=10\n");
    }
    
    // Create tableau
    Tableau* tab = createTableau(lp);
    
    // Print initial tableau
    printf("\nInitial Tableau:\n");
    printTableau(tab);
    
    // Solve
    SimplexStatus status = solveSimplex(tab, lp);
    
    // Print solution
    printSolution(tab, lp, status);
    
    // Cleanup
    freeTableau(tab);
    freeLPProblem(lp);
    
    return (status == OPTIMAL) ? EXIT_SUCCESS : EXIT_FAILURE;
}
