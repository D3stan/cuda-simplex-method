#include "solver.h"

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
    __shared__ double sharedPivotElem[BLOCK_SIZE];  // Track pivot element for Harris test
    
    int tid = threadIdx.x;
    
    // Initialize with "infinity"
    sharedRatio[tid] = DBL_MAX;
    sharedRow[tid] = -1;
    sharedPivotElem[tid] = 0.0;
    
    // Each thread processes one or more rows (skip row 0 = objective)
    // Harris ratio test: among near-tied ratios, prefer largest pivot element
    for (int row = tid + 1; row < numRows; row += blockDim.x) {
        double aij = tableau[row * numCols + pivotCol];
        double bi = tableau[row * numCols + (numCols - 1)];  // RHS column
        
        // Minimum ratio test: require pivot element above PIVOT_TOL for stability
        if (aij > PIVOT_TOL) {
            double ratio = bi / aij;
            if (ratio >= -EPSILON) {
                if (ratio < sharedRatio[tid] - HARRIS_TOL) {
                    // Clearly better ratio
                    sharedRatio[tid] = ratio;
                    sharedRow[tid] = row;
                    sharedPivotElem[tid] = aij;
                } else if (ratio < sharedRatio[tid] + HARRIS_TOL && aij > sharedPivotElem[tid]) {
                    // Similar ratio but larger pivot element — prefer for stability
                    sharedRatio[tid] = ratio;
                    sharedRow[tid] = row;
                    sharedPivotElem[tid] = aij;
                }
            }
        }
    }
    __syncthreads();
    
    // Reduction to find best pivot row (Harris ratio test)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedRatio[tid + stride] < sharedRatio[tid] - HARRIS_TOL) {
                // Clearly smaller ratio wins
                sharedRatio[tid] = sharedRatio[tid + stride];
                sharedRow[tid] = sharedRow[tid + stride];
                sharedPivotElem[tid] = sharedPivotElem[tid + stride];
            } else if (sharedRatio[tid + stride] < sharedRatio[tid] + HARRIS_TOL) {
                // Ratios are close — prefer larger pivot element for numerical stability
                if (sharedPivotElem[tid + stride] > sharedPivotElem[tid]) {
                    sharedRatio[tid] = sharedRatio[tid + stride];
                    sharedRow[tid] = sharedRow[tid + stride];
                    sharedPivotElem[tid] = sharedPivotElem[tid + stride];
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

