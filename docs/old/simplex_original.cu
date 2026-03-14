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

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>

// ===========================================================================
// CONSTANTS AND CONFIGURATION
// ===========================================================================

#define EPSILON 1e-10
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Global verbosity flag (0 = silent, 1 = normal, 2 = diagnostic only)
static int g_verbose = 1;

// Output format
typedef enum { OUTPUT_TEXT, OUTPUT_JSON, OUTPUT_CSV } OutputFormat;
static OutputFormat g_outputFormat = OUTPUT_TEXT;

// Iteration log file (NULL = disabled)
static FILE* g_iterLog = NULL;

// Current phase and total iteration counter (for logging)
static int g_phase = 0;
static int g_totalIterations = 0;

// Debug flag: print initial, intermediate, and final tableaux
static int g_debug = 0;

// Max iterations (overridable via -m)
static int g_maxIter = 50000;

// Timeout in seconds (0 = no timeout)
static double g_timeout = 0.0;
static double g_solveStartTime = 0.0;

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
    TIMEOUT,
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
    
    double* lowerBounds;   // Variable lower bounds (default 0.0)
    double* upperBounds;   // Variable upper bounds (default DBL_MAX)
    double objConstant;    // Constant term in objective (from RHS of obj row)
    int* isInteger;        // 1 if variable is integer, 0 otherwise
    double* rangeValues;   // Range values for constraints (0 if no range)
    
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

// Strip trailing newline/CR only (preserve column positions for fixed-format)
static void stripNewline(char* str) {
    int len = (int)strlen(str);
    while (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
        str[--len] = '\0';
    }
}

/**
 * Extract a field from a fixed-column MPS line.
 * startCol and endCol are 0-indexed inclusive.
 * Only trailing whitespace is trimmed; leading spaces are preserved
 * because MPS names can legally contain embedded/leading spaces.
 */
static void extractMPSField(const char* line, int startCol, int endCol,
                             char* out, int outSize) {
    int lineLen = (int)strlen(line);
    int i = 0;
    for (int c = startCol; c <= endCol && c < lineLen && i < outSize - 1; c++) {
        out[i++] = line[c];
    }
    out[i] = '\0';
    // Trim trailing whitespace only
    while (i > 0 && (out[i-1] == ' ' || out[i-1] == '\t')) {
        out[--i] = '\0';
    }
}

// Parse a double from a string field; returns 1 on success, 0 on failure
static int parseMPSDouble(const char* field, double* value) {
    if (field[0] == '\0') return 0;
    char* endptr;
    *value = strtod(field, &endptr);
    while (*endptr == ' ' || *endptr == '\t') endptr++;
    return (endptr != field && *endptr == '\0');
}

// Check if line is a section header (non-space at column 0, not a comment)
static int isMPSSectionHeader(const char* line) {
    return (line[0] != '\0' && line[0] != ' ' && line[0] != '\t' && line[0] != '*');
}

/**
 * Parse an MPS file into an LPProblem.
 *
 * Single-pass, fixed-column format, dynamic allocation.
 * Handles: ROWS, COLUMNS, RHS, BOUNDS, RANGES, OBJSENSE, integer MARKERs.
 * Stops strictly at ENDATA.
 */
LPProblem* parseMPS(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    LPProblem* lp = (LPProblem*)calloc(1, sizeof(LPProblem));
    lp->sense = MINIMIZE;
    lp->objConstant = 0.0;
    
    char rawLine[1024];
    char section[64] = "";
    
    // --- Dynamic storage for rows ---
    int rowCap = 64;
    int numRows = 0;
    char** rowNames = (char**)malloc(rowCap * sizeof(char*));
    ConstraintType* rowTypes = (ConstraintType*)malloc(rowCap * sizeof(ConstraintType));
    double* rhsValues = (double*)calloc(rowCap, sizeof(double));
    double* rangeVals = (double*)calloc(rowCap, sizeof(double));
    char* objRowName = NULL;
    
    // --- Dynamic storage for variables ---
    int varCap = 64;
    int numVars = 0;
    typedef struct { char name[64]; } VarName;
    VarName* varNamesBuf = (VarName*)malloc(varCap * sizeof(VarName));
    double* objCoeffsTemp = (double*)calloc(varCap, sizeof(double));
    double* loBounds = (double*)calloc(varCap, sizeof(double));   // default 0
    double* upBounds = (double*)malloc(varCap * sizeof(double));
    int* isInt = (int*)calloc(varCap, sizeof(int));
    for (int i = 0; i < varCap; i++) upBounds[i] = DBL_MAX;
    
    // --- Dynamic storage for sparse coefficients ---
    int coeffCap = 256;
    int numCoeffs = 0;
    typedef struct { int row; int col; double value; } CoeffEntry;
    CoeffEntry* coeffs = (CoeffEntry*)malloc(coeffCap * sizeof(CoeffEntry));
    
    // Integer marker state for COLUMNS section
    int inIntegerBlock = 0;
    
    // --- Helper: grow variable arrays when capacity is exceeded ---
    auto growVarArrays = [&]() {
        int oldCap = varCap;
        varCap *= 2;
        varNamesBuf   = (VarName*)realloc(varNamesBuf, varCap * sizeof(VarName));
        objCoeffsTemp = (double*)realloc(objCoeffsTemp, varCap * sizeof(double));
        loBounds      = (double*)realloc(loBounds, varCap * sizeof(double));
        upBounds      = (double*)realloc(upBounds, varCap * sizeof(double));
        isInt         = (int*)realloc(isInt, varCap * sizeof(int));
        for (int i = oldCap; i < varCap; i++) {
            objCoeffsTemp[i] = 0.0;
            loBounds[i] = 0.0;
            upBounds[i] = DBL_MAX;
            isInt[i] = 0;
        }
    };
    
    // --- Helper: grow row arrays when capacity is exceeded ---
    auto growRowArrays = [&]() {
        int oldCap = rowCap;
        rowCap *= 2;
        rowNames  = (char**)realloc(rowNames, rowCap * sizeof(char*));
        rowTypes  = (ConstraintType*)realloc(rowTypes, rowCap * sizeof(ConstraintType));
        rhsValues = (double*)realloc(rhsValues, rowCap * sizeof(double));
        rangeVals = (double*)realloc(rangeVals, rowCap * sizeof(double));
        for (int i = oldCap; i < rowCap; i++) {
            rhsValues[i] = 0.0;
            rangeVals[i] = 0.0;
        }
    };
    
    // --- Helper: find variable index by name ---
    auto findVar = [&](const char* name) -> int {
        for (int i = 0; i < numVars; i++) {
            if (strcmp(varNamesBuf[i].name, name) == 0) return i;
        }
        return -1;
    };
    
    // --- Helper: add a new variable, returns its index ---
    auto addVar = [&](const char* name) -> int {
        if (numVars >= varCap) growVarArrays();
        strncpy(varNamesBuf[numVars].name, name, 63);
        varNamesBuf[numVars].name[63] = '\0';
        objCoeffsTemp[numVars] = 0.0;
        loBounds[numVars] = 0.0;
        upBounds[numVars] = DBL_MAX;
        isInt[numVars] = inIntegerBlock ? 1 : 0;
        return numVars++;
    };
    
    // --- Helper: find row index (-1 = objective, -2 = not found) ---
    auto findRow = [&](const char* name) -> int {
        if (objRowName && strcmp(name, objRowName) == 0) return -1;
        for (int i = 0; i < numRows; i++) {
            if (strcmp(rowNames[i], name) == 0) return i;
        }
        return -2;
    };
    
    // --- Helper: add a sparse coefficient entry ---
    auto addCoeff = [&](int row, int col, double val) {
        if (numCoeffs >= coeffCap) {
            coeffCap *= 2;
            coeffs = (CoeffEntry*)realloc(coeffs, coeffCap * sizeof(CoeffEntry));
        }
        coeffs[numCoeffs].row = row;
        coeffs[numCoeffs].col = col;
        coeffs[numCoeffs].value = val;
        numCoeffs++;
    };
    
    // ===================== MAIN PARSE LOOP =====================
    while (fgets(rawLine, sizeof(rawLine), file)) {
        stripNewline(rawLine);
        
        // Skip empty lines and comments (* in column 1)
        if (rawLine[0] == '\0') continue;
        if (rawLine[0] == '*') continue;
        
        // --- Section headers start at column 1 (non-space, non-comment) ---
        if (isMPSSectionHeader(rawLine)) {
            if (strncmp(rawLine, "NAME", 4) == 0) {
                // Name field starts at column 14 (0-indexed)
                if ((int)strlen(rawLine) > 14) {
                    extractMPSField(rawLine, 14, 71, lp->name, sizeof(lp->name));
                }
                continue;
            }
            if (strncmp(rawLine, "ROWS", 4) == 0)     { strcpy(section, "ROWS"); continue; }
            if (strncmp(rawLine, "COLUMNS", 7) == 0)   { strcpy(section, "COLUMNS"); continue; }
            if (strncmp(rawLine, "RHS", 3) == 0)       { strcpy(section, "RHS"); continue; }
            if (strncmp(rawLine, "BOUNDS", 6) == 0)     { strcpy(section, "BOUNDS"); continue; }
            if (strncmp(rawLine, "RANGES", 6) == 0)     { strcpy(section, "RANGES"); continue; }
            if (strncmp(rawLine, "ENDATA", 6) == 0)     break;  // *** STOP parsing ***
            if (strncmp(rawLine, "OBJSENSE", 8) == 0) {
                if (fgets(rawLine, sizeof(rawLine), file)) {
                    stripNewline(rawLine);
                    char senseBuf[16];
                    extractMPSField(rawLine, 0, 15, senseBuf, sizeof(senseBuf));
                    char* s = senseBuf;
                    while (*s == ' ' || *s == '\t') s++;
                    if (strncmp(s, "MAX", 3) == 0) lp->sense = MAXIMIZE;
                    else if (strncmp(s, "MIN", 3) == 0) lp->sense = MINIMIZE;
                }
                continue;
            }
            // Unknown section — record name and skip its data lines
            extractMPSField(rawLine, 0, 15, section, sizeof(section));
            continue;
        }
        
        // --- Data lines (start with space/tab) ---
        // MPS fixed-column fields (0-indexed):
        //   Field 1: cols  1- 2  (type indicator)
        //   Field 2: cols  4-11  (name 1, 8 chars)
        //   Field 3: cols 14-21  (name 2, 8 chars)
        //   Field 4: cols 24-35  (value 1, 12 chars)
        //   Field 5: cols 39-46  (name 3, 8 chars)
        //   Field 6: cols 49-60  (value 2, 12 chars)
        
        // ---- ROWS section ----
        if (strcmp(section, "ROWS") == 0) {
            char typeField[4], nameField[64];
            extractMPSField(rawLine, 1, 2, typeField, sizeof(typeField));
            extractMPSField(rawLine, 4, 11, nameField, sizeof(nameField));
            
            if (nameField[0] == '\0') continue;
            
            char type = typeField[0];
            if (type == 'N') {
                // Only the first N row is used as the objective (per MPS convention)
                if (!objRowName) {
                    objRowName = strdup(nameField);
                } else {
                    fprintf(stderr, "Warning: Multiple N rows found. "
                            "Using '%s' as objective, ignoring '%s'.\n",
                            objRowName, nameField);
                }
            } else {
                if (numRows >= rowCap) growRowArrays();
                rowNames[numRows] = strdup(nameField);
                switch (type) {
                    case 'L': rowTypes[numRows] = CONSTRAINT_LE; break;
                    case 'G': rowTypes[numRows] = CONSTRAINT_GE; break;
                    case 'E': rowTypes[numRows] = CONSTRAINT_EQ; break;
                    default:
                        fprintf(stderr, "Warning: Unknown row type '%c' for '%s', "
                                "defaulting to EQ.\n", type, nameField);
                        rowTypes[numRows] = CONSTRAINT_EQ;
                }
                numRows++;
            }
        }
        // ---- COLUMNS section ----
        else if (strcmp(section, "COLUMNS") == 0) {
            char field2[64], field3[64], field4[64], field5[64], field6[64];
            extractMPSField(rawLine,  4, 11, field2, sizeof(field2));  // column name
            extractMPSField(rawLine, 14, 21, field3, sizeof(field3));  // row name 1
            extractMPSField(rawLine, 24, 35, field4, sizeof(field4));  // value 1
            extractMPSField(rawLine, 39, 46, field5, sizeof(field5));  // row name 2
            extractMPSField(rawLine, 49, 60, field6, sizeof(field6));  // value 2
            
            if (field2[0] == '\0') continue;
            
            // Handle integer MARKER lines (INTORG / INTEND)
            if (strcmp(field3, "'MARKER'") == 0 || strcmp(field3, "MARKER") == 0) {
                if (strstr(field4, "INTORG") || strstr(field5, "INTORG") ||
                    strstr(field4, "'INTORG'") || strstr(field5, "'INTORG'")) {
                    inIntegerBlock = 1;
                } else if (strstr(field4, "INTEND") || strstr(field5, "INTEND") ||
                           strstr(field4, "'INTEND'") || strstr(field5, "'INTEND'")) {
                    inIntegerBlock = 0;
                }
                continue;
            }
            
            double val1;
            if (!parseMPSDouble(field4, &val1)) continue;
            
            // Find or create variable
            int varIdx = findVar(field2);
            if (varIdx < 0) varIdx = addVar(field2);
            
            // First coefficient
            int rowIdx = findRow(field3);
            if (rowIdx == -1) {
                objCoeffsTemp[varIdx] = val1;           // Objective
            } else if (rowIdx >= 0) {
                addCoeff(rowIdx, varIdx, val1);         // Constraint
            }
            
            // Second coefficient (fields 5+6, optional)
            double val2;
            if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
                rowIdx = findRow(field5);
                if (rowIdx == -1) {
                    objCoeffsTemp[varIdx] = val2;
                } else if (rowIdx >= 0) {
                    addCoeff(rowIdx, varIdx, val2);
                }
            }
        }
        // ---- RHS section ----
        else if (strcmp(section, "RHS") == 0) {
            char field2[64], field3[64], field4[64], field5[64], field6[64];
            extractMPSField(rawLine,  4, 11, field2, sizeof(field2));
            extractMPSField(rawLine, 14, 21, field3, sizeof(field3));
            extractMPSField(rawLine, 24, 35, field4, sizeof(field4));
            extractMPSField(rawLine, 39, 46, field5, sizeof(field5));
            extractMPSField(rawLine, 49, 60, field6, sizeof(field6));
            
            double val1;
            if (field3[0] != '\0' && parseMPSDouble(field4, &val1)) {
                int rowIdx = findRow(field3);
                if (rowIdx >= 0) {
                    rhsValues[rowIdx] = val1;
                } else if (rowIdx == -1) {
                    // Objective constant (RHS entry for the N row)
                    lp->objConstant = val1;
                }
            }
            
            double val2;
            if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
                int rowIdx = findRow(field5);
                if (rowIdx >= 0) {
                    rhsValues[rowIdx] = val2;
                } else if (rowIdx == -1) {
                    lp->objConstant = val2;
                }
            }
        }
        // ---- BOUNDS section ----
        else if (strcmp(section, "BOUNDS") == 0) {
            // Field 1 (cols 1-2): bound type  (LO, UP, FX, FR, MI, PL, BV, LI, UI)
            // Field 2 (cols 4-11): bound set name
            // Field 3 (cols 14-21): variable name
            // Field 4 (cols 24-35): value (absent for FR, MI, PL, BV)
            char typeField[4], field2[64], field3[64], field4[64];
            extractMPSField(rawLine,  1,  2, typeField, sizeof(typeField));
            extractMPSField(rawLine,  4, 11, field2, sizeof(field2));
            extractMPSField(rawLine, 14, 21, field3, sizeof(field3));
            extractMPSField(rawLine, 24, 35, field4, sizeof(field4));
            
            if (field3[0] == '\0') continue;
            
            int varIdx = findVar(field3);
            if (varIdx < 0) {
                fprintf(stderr, "Warning: BOUNDS references unknown variable '%s'.\n",
                        field3);
                continue;
            }
            
            double val = 0.0;
            int hasValue = parseMPSDouble(field4, &val);
            
            if (strcmp(typeField, "LO") == 0 && hasValue) {
                loBounds[varIdx] = val;
            } else if (strcmp(typeField, "UP") == 0 && hasValue) {
                upBounds[varIdx] = val;
            } else if (strcmp(typeField, "FX") == 0 && hasValue) {
                loBounds[varIdx] = val;
                upBounds[varIdx] = val;
            } else if (strcmp(typeField, "FR") == 0) {
                loBounds[varIdx] = -DBL_MAX;
                upBounds[varIdx] = DBL_MAX;
            } else if (strcmp(typeField, "MI") == 0) {
                loBounds[varIdx] = -DBL_MAX;
            } else if (strcmp(typeField, "PL") == 0) {
                upBounds[varIdx] = DBL_MAX;
            } else if (strcmp(typeField, "BV") == 0) {
                loBounds[varIdx] = 0.0;
                upBounds[varIdx] = 1.0;
                isInt[varIdx] = 1;
            } else if (strcmp(typeField, "LI") == 0 && hasValue) {
                loBounds[varIdx] = val;
                isInt[varIdx] = 1;
            } else if (strcmp(typeField, "UI") == 0 && hasValue) {
                upBounds[varIdx] = val;
                isInt[varIdx] = 1;
            } else {
                fprintf(stderr, "Warning: Unknown/invalid bound type '%s' "
                        "for variable '%s'.\n", typeField, field3);
            }
        }
        // ---- RANGES section ----
        else if (strcmp(section, "RANGES") == 0) {
            // Same field layout as RHS: name, row, value [, row, value]
            char field2[64], field3[64], field4[64], field5[64], field6[64];
            extractMPSField(rawLine,  4, 11, field2, sizeof(field2));
            extractMPSField(rawLine, 14, 21, field3, sizeof(field3));
            extractMPSField(rawLine, 24, 35, field4, sizeof(field4));
            extractMPSField(rawLine, 39, 46, field5, sizeof(field5));
            extractMPSField(rawLine, 49, 60, field6, sizeof(field6));
            
            double val1;
            if (field3[0] != '\0' && parseMPSDouble(field4, &val1)) {
                int rowIdx = findRow(field3);
                if (rowIdx >= 0) rangeVals[rowIdx] = val1;
            }
            
            double val2;
            if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
                int rowIdx = findRow(field5);
                if (rowIdx >= 0) rangeVals[rowIdx] = val2;
            }
        }
        // Unknown section: silently skip data lines
    }
    
    fclose(file);
    
    // ===================== BUILD LPProblem STRUCTURE =====================
    lp->numVars = numVars;
    lp->numConstraints = numRows;
    
    lp->objCoeffs = (double*)malloc(numVars * sizeof(double));
    memcpy(lp->objCoeffs, objCoeffsTemp, numVars * sizeof(double));
    
    lp->lowerBounds = (double*)malloc(numVars * sizeof(double));
    memcpy(lp->lowerBounds, loBounds, numVars * sizeof(double));
    
    lp->upperBounds = (double*)malloc(numVars * sizeof(double));
    memcpy(lp->upperBounds, upBounds, numVars * sizeof(double));
    
    lp->isInteger = (int*)malloc(numVars * sizeof(int));
    memcpy(lp->isInteger, isInt, numVars * sizeof(int));
    
    lp->rhs = (double*)malloc(numRows * sizeof(double));
    lp->constraintTypes = (ConstraintType*)malloc(numRows * sizeof(ConstraintType));
    lp->constraintMatrix = (double**)malloc(numRows * sizeof(double*));
    lp->varNames = (char**)malloc(numVars * sizeof(char*));
    lp->constraintNames = (char**)malloc(numRows * sizeof(char*));
    lp->rangeValues = (double*)malloc(numRows * sizeof(double));
    
    for (int i = 0; i < numRows; i++) {
        lp->constraintMatrix[i] = (double*)calloc(numVars, sizeof(double));
        lp->rhs[i] = rhsValues[i];
        lp->constraintTypes[i] = rowTypes[i];
        lp->constraintNames[i] = rowNames[i];   // Transfer ownership
        lp->rangeValues[i] = rangeVals[i];
    }
    
    for (int i = 0; i < numVars; i++) {
        lp->varNames[i] = strdup(varNamesBuf[i].name);
    }
    
    // Fill constraint matrix from sparse coefficients
    for (int i = 0; i < numCoeffs; i++) {
        lp->constraintMatrix[coeffs[i].row][coeffs[i].col] = coeffs[i].value;
    }
    
    // Cleanup temporary storage
    free(rangeVals);
    free(rhsValues);
    free(coeffs);
    free(isInt);
    free(upBounds);
    free(loBounds);
    free(objCoeffsTemp);
    free(varNamesBuf);
    free(rowTypes);
    free(rowNames);   // Individual names transferred; only free the pointer array
    if (objRowName) free(objRowName);
    
    if (g_verbose) {
        printf("Parsed LP: %s\n", lp->name);
        printf("  Variables: %d\n", lp->numVars);
        printf("  Constraints: %d\n", lp->numConstraints);
        printf("  Sense: %s\n", lp->sense == MAXIMIZE ? "MAXIMIZE" : "MINIMIZE");
        if (lp->objConstant != 0.0)
            printf("  Objective constant: %.6f\n", lp->objConstant);
    }
    
    return lp;
}

/**
 * Preprocess variable bounds and range constraints.
 *
 * Transforms the LP into a form suitable for the standard simplex:
 *   1. Shift variables with finite non-zero lower bounds (x' = x - lo)
 *   2. Split free variables (lo = -inf) into x = x+ - x-
 *   3. Add upper-bound constraints (x_j <= ub)
 *   4. Expand range constraints into pairs of inequalities
 */
void preprocessBounds(LPProblem* lp) {
    if (!lp->lowerBounds || !lp->upperBounds) return;
    
    int origVars = lp->numVars;
    int origConstraints = lp->numConstraints;
    
    // --- Step 1: Shift variables with finite non-zero lower bounds ---
    for (int j = 0; j < origVars; j++) {
        double lo = lp->lowerBounds[j];
        if (lo != 0.0 && lo > -DBL_MAX) {
            // x_j = x_j' + lo  =>  a_ij * x_j = a_ij * x_j' + a_ij * lo
            for (int i = 0; i < origConstraints; i++) {
                lp->rhs[i] -= lp->constraintMatrix[i][j] * lo;
            }
            lp->objConstant += lp->objCoeffs[j] * lo;
            if (lp->upperBounds[j] < DBL_MAX)
                lp->upperBounds[j] -= lo;
            lp->lowerBounds[j] = 0.0;
        }
    }
    
    // --- Count extra variables (free-var splits) and extra constraints ---
    int numFreeVars = 0;
    int numUpperBounds = 0;
    int numRanges = 0;
    
    for (int j = 0; j < origVars; j++) {
        if (lp->lowerBounds[j] <= -DBL_MAX) numFreeVars++;
        if (lp->upperBounds[j] < DBL_MAX)   numUpperBounds++;
    }
    if (lp->rangeValues) {
        for (int i = 0; i < origConstraints; i++) {
            if (lp->rangeValues[i] != 0.0) numRanges++;
        }
    }
    
    int extraVars = numFreeVars;
    int extraConstraints = numUpperBounds + numRanges;
    if (extraVars == 0 && extraConstraints == 0) return;
    
    int newNumVars = origVars + extraVars;
    int newNumConstraints = origConstraints + extraConstraints;
    
    // --- Step 2: Split free variables (lo = -inf) into x+ - x- ---
    if (extraVars > 0) {
        lp->objCoeffs   = (double*)realloc(lp->objCoeffs, newNumVars * sizeof(double));
        lp->lowerBounds  = (double*)realloc(lp->lowerBounds, newNumVars * sizeof(double));
        lp->upperBounds  = (double*)realloc(lp->upperBounds, newNumVars * sizeof(double));
        lp->isInteger    = (int*)realloc(lp->isInteger, newNumVars * sizeof(int));
        lp->varNames     = (char**)realloc(lp->varNames, newNumVars * sizeof(char*));
        
        for (int i = 0; i < origConstraints; i++) {
            lp->constraintMatrix[i] = (double*)realloc(
                lp->constraintMatrix[i], newNumVars * sizeof(double));
            for (int k = origVars; k < newNumVars; k++)
                lp->constraintMatrix[i][k] = 0.0;
        }
        
        int nextVar = origVars;
        for (int j = 0; j < origVars; j++) {
            if (lp->lowerBounds[j] <= -DBL_MAX) {
                int neg = nextVar++;
                for (int i = 0; i < origConstraints; i++)
                    lp->constraintMatrix[i][neg] = -lp->constraintMatrix[i][j];
                lp->objCoeffs[neg]  = -lp->objCoeffs[j];
                lp->lowerBounds[j]  = 0.0;
                lp->upperBounds[j]  = DBL_MAX;
                lp->lowerBounds[neg] = 0.0;
                lp->upperBounds[neg] = DBL_MAX;
                lp->isInteger[neg]  = 0;
                
                char buf[128];
                snprintf(buf, sizeof(buf), "%s_neg", lp->varNames[j]);
                lp->varNames[neg] = strdup(buf);
            }
        }
        lp->numVars = newNumVars;
    }
    
    // --- Step 3 & 4: Add upper-bound and range constraints ---
    if (extraConstraints > 0) {
        lp->rhs             = (double*)realloc(lp->rhs, newNumConstraints * sizeof(double));
        lp->constraintTypes  = (ConstraintType*)realloc(lp->constraintTypes,
                                    newNumConstraints * sizeof(ConstraintType));
        lp->constraintMatrix = (double**)realloc(lp->constraintMatrix,
                                    newNumConstraints * sizeof(double*));
        lp->constraintNames  = (char**)realloc(lp->constraintNames,
                                    newNumConstraints * sizeof(char*));
        lp->rangeValues      = (double*)realloc(lp->rangeValues,
                                    newNumConstraints * sizeof(double));
        
        int nextRow = origConstraints;
        
        // Upper-bound constraints: x_j <= UB
        for (int j = 0; j < origVars; j++) {
            if (lp->upperBounds[j] < DBL_MAX) {
                int r = nextRow++;
                lp->constraintMatrix[r] = (double*)calloc(lp->numVars, sizeof(double));
                lp->constraintMatrix[r][j] = 1.0;
                lp->rhs[r] = lp->upperBounds[j];
                lp->constraintTypes[r] = CONSTRAINT_LE;
                lp->rangeValues[r] = 0.0;
                
                char buf[128];
                snprintf(buf, sizeof(buf), "_UB_%s", lp->varNames[j]);
                lp->constraintNames[r] = strdup(buf);
            }
        }
        
        // Range constraints: add the complementary inequality
        for (int i = 0; i < origConstraints; i++) {
            if (lp->rangeValues && lp->rangeValues[i] != 0.0) {
                double rv = lp->rangeValues[i];
                int r = nextRow++;
                lp->constraintMatrix[r] = (double*)calloc(lp->numVars, sizeof(double));
                memcpy(lp->constraintMatrix[r], lp->constraintMatrix[i],
                       lp->numVars * sizeof(double));
                lp->rangeValues[r] = 0.0;
                
                char buf[128];
                snprintf(buf, sizeof(buf), "_RNG_%s", lp->constraintNames[i]);
                lp->constraintNames[r] = strdup(buf);
                
                double origRhs = lp->rhs[i];
                switch (lp->constraintTypes[i]) {
                    case CONSTRAINT_GE:
                        // Keep >= b, add <= b + |r|
                        lp->constraintTypes[r] = CONSTRAINT_LE;
                        lp->rhs[r] = origRhs + fabs(rv);
                        break;
                    case CONSTRAINT_LE:
                        // Keep <= b, add >= b - |r|
                        lp->constraintTypes[r] = CONSTRAINT_GE;
                        lp->rhs[r] = origRhs - fabs(rv);
                        break;
                    case CONSTRAINT_EQ:
                        if (rv > 0) {
                            // h = b, u = b + |r|  =>  >= b  AND  <= b + |r|
                            lp->constraintTypes[i] = CONSTRAINT_GE;
                            lp->constraintTypes[r] = CONSTRAINT_LE;
                            lp->rhs[r] = origRhs + fabs(rv);
                        } else {
                            // h = b - |r|, u = b  =>  >= b - |r|  AND  <= b
                            lp->constraintTypes[i] = CONSTRAINT_GE;
                            lp->rhs[i] = origRhs - fabs(rv);
                            lp->constraintTypes[r] = CONSTRAINT_LE;
                            lp->rhs[r] = origRhs;
                        }
                        break;
                }
                lp->rangeValues[i] = 0.0;  // Consumed
            }
        }
        
        lp->numConstraints = newNumConstraints;
    }
    
    if (g_verbose) {
        printf("After bound preprocessing:\n");
        printf("  Variables: %d (was %d)\n", lp->numVars, origVars);
        printf("  Constraints: %d (was %d)\n", lp->numConstraints, origConstraints);
    }
}

void freeLPProblem(LPProblem* lp) {
    if (!lp) return;
    
    free(lp->objCoeffs);
    free(lp->rhs);
    free(lp->constraintTypes);
    free(lp->lowerBounds);
    free(lp->upperBounds);
    free(lp->isInteger);
    free(lp->rangeValues);
    
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
        tab->hostData[j] = 1e10;  // Big-M
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

static int extractArtificialBasis(Tableau* tab) {
    syncTableauToHost(tab);

    int changedAny = 0;
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
                changedAny = 1;
                if (g_verbose >= 2) {
                    printf("[DIAG] Extracted artificial var via pivot: row %d, enter col %d\n",
                           row, entering);
                }
                break;
            }

            double rhsVal = tab->hostData[row * tab->cols + rhsCol];
            if (fabs(rhsVal) > 1e-8) {
                if (g_verbose) {
                    printf("Error: Artificial variable in basis has non-zero value (row %d, rhs %.6e)\n",
                           row, rhsVal);
                }
                return -1;
            }

            if (!eliminateDegenerateArtificialRowCol(tab, i, basicVar)) return -1;
            changedThisPass = 1;
            changedAny = 1;
            if (g_verbose >= 2) {
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

    return changedAny;
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
            if (g_verbose)
                printf("Iteration %d: Problem is unbounded (no valid pivot row for column %d)\n",
                       iteration, h_pivotCol);
            status = UNBOUNDED;
            break;
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
        
        g_phase = 1;
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
        
        if (g_verbose) printf("Phase 1 objective value: %.10f\n", phase1Obj);
        
        if (fabs(phase1Obj) > EPSILON) {
            if (g_verbose) printf("Problem is INFEASIBLE (Phase 1 objective = %.6f)\n", phase1Obj);
            free(originalObjective);
            return INFEASIBLE;
        }
        
        // Extract artificial variables from basis when possible (degenerate case).
        // If a basic artificial row has a non-zero coefficient in any non-artificial
        // column, pivot it out; otherwise (all-zero non-artificial row with RHS=0),
        // remove that row and its artificial column from the tableau.
        int extractStatus = extractArtificialBasis(tab);
        if (extractStatus < 0) {
            if (g_verbose) printf("Error: Failed to extract degenerate artificial basis\n");
            free(originalObjective);
            return ERROR;
        }
        if (extractStatus > 0 && g_verbose) {
            printf("Artificial basis cleanup completed before Phase 2\n");
        }
        
        if (g_verbose) printf("\n--- Phase 2: Optimizing original objective ---\n");
        setupPhase2(tab, originalObjective);
        
        free(originalObjective);
    } else {
        if (g_verbose) printf("\nNo artificial variables needed - direct optimization\n");
    }
    
    // Phase 2 (or single phase if no artificials)
    g_phase = (tab->numArtificial > 0) ? 2 : 0;
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

// ===========================================================================
// OUTPUT FORMATTERS (JSON / CSV)
// ===========================================================================

/**
 * Helper: extract solution values and compute objective
 */
static double extractSolutionValues(Tableau* tab, LPProblem* lp, double** outSolution) {
    syncTableauToHost(tab);
    
    double* solution = (double*)calloc(tab->numOriginalVars, sizeof(double));
    
    for (int i = 0; i < tab->rows - 1; i++) {
        int basicVar = tab->hostBasicVars[i];
        if (basicVar < tab->numOriginalVars) {
            solution[basicVar] = tab->hostData[(i + 1) * tab->cols + (tab->cols - 1)];
        }
    }
    
    double objValue = lp->objConstant;
    for (int i = 0; i < lp->numVars; i++) {
        objValue += lp->objCoeffs[i] * solution[i];
    }
    
    *outSolution = solution;
    return objValue;
}

static const char* statusString(SimplexStatus status) {
    switch (status) {
        case OPTIMAL:    return "OPTIMAL";
        case INFEASIBLE: return "INFEASIBLE";
        case UNBOUNDED:  return "UNBOUNDED";
        case TIMEOUT:    return "TIMEOUT";
        case ERROR:      return "ERROR";
        default:         return "UNKNOWN";
    }
}

/**
 * Print solution in JSON format
 */
void printSolutionJSON(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed) {
    printf("{\n");
    printf("  \"problem\": \"%s\",\n", lp->name);
    printf("  \"status\": \"%s\",\n", statusString(status));
    printf("  \"variables\": %d,\n", lp->numVars);
    printf("  \"constraints\": %d,\n", lp->numConstraints);
    printf("  \"sense\": \"%s\",\n", lp->sense == MAXIMIZE ? "maximize" : "minimize");
    printf("  \"iterations\": %d,\n", g_totalIterations);
    printf("  \"elapsed_seconds\": %.6f", elapsed);
    
    if (status == OPTIMAL) {
        double* solution;
        double objValue = extractSolutionValues(tab, lp, &solution);
        
        printf(",\n  \"objective_value\": %.10f,\n", objValue);
        if (lp->objConstant != 0.0)
            printf("  \"objective_constant\": %.10f,\n", lp->objConstant);
        printf("  \"solution\": {");
        int first = 1;
        for (int i = 0; i < lp->numVars; i++) {
            if (fabs(solution[i]) > EPSILON) {
                if (!first) printf(",");
                printf("\n    \"%s\": %.10f", lp->varNames[i], solution[i]);
                first = 0;
            }
        }
        printf("\n  }");
        free(solution);
    }
    printf("\n}\n");
}

/**
 * Print solution in CSV format (one header + one data row)
 */
void printSolutionCSV(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed) {
    printf("problem,status,variables,constraints,sense,iterations,elapsed_seconds,objective_value\n");
    
    double objValue = 0.0;
    if (status == OPTIMAL) {
        double* solution;
        objValue = extractSolutionValues(tab, lp, &solution);
        free(solution);
    }
    
    printf("%s,%s,%d,%d,%s,%d,%.6f,%.10f\n",
           lp->name, statusString(status),
           lp->numVars, lp->numConstraints,
           lp->sense == MAXIMIZE ? "maximize" : "minimize",
           g_totalIterations, elapsed, objValue);
}

/**
 * Dispatch solution output based on g_outputFormat
 */
void outputSolution(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed) {
    switch (g_outputFormat) {
        case OUTPUT_JSON:
            printSolutionJSON(tab, lp, status, elapsed);
            break;
        case OUTPUT_CSV:
            printSolutionCSV(tab, lp, status, elapsed);
            break;
        case OUTPUT_TEXT:
        default:
            printSolution(tab, lp, status);
            printf("\nElapsed time: %.6f seconds\n", elapsed);
            printf("Total iterations: %d\n", g_totalIterations);
            break;
    }
}

// ===========================================================================
// BATCH MODE
// ===========================================================================

typedef struct {
    char filename[512];
    int numVars;
    int numConstraints;
    const char* statusStr;
    double objValue;
    int iterations;
    double elapsed;
} BatchResult;

void printBatchSummaryText(BatchResult* results, int count) {
    printf("\n%-20s %6s %6s  %-12s %18s %8s %10s\n",
           "Problem", "Vars", "Cons", "Status", "Obj Value", "Iters", "Time(s)");
    printf("%-20s %6s %6s  %-12s %18s %8s %10s\n",
           "--------------------", "------", "------", "------------",
           "------------------", "--------", "----------");
    
    for (int i = 0; i < count; i++) {
        printf("%-20s %6d %6d  %-12s %18.6e %8d %10.4f\n",
               results[i].filename, results[i].numVars, results[i].numConstraints,
               results[i].statusStr, results[i].objValue,
               results[i].iterations, results[i].elapsed);
    }
}

void printBatchSummaryJSON(BatchResult* results, int count) {
    printf("[\n");
    for (int i = 0; i < count; i++) {
        printf("  {\"problem\": \"%s\", \"variables\": %d, \"constraints\": %d, "
               "\"status\": \"%s\", \"objective_value\": %.10f, "
               "\"iterations\": %d, \"elapsed_seconds\": %.6f}%s\n",
               results[i].filename, results[i].numVars, results[i].numConstraints,
               results[i].statusStr, results[i].objValue,
               results[i].iterations, results[i].elapsed,
               (i < count - 1) ? "," : "");
    }
    printf("]\n");
}

void printBatchSummaryCSV(BatchResult* results, int count) {
    printf("problem,variables,constraints,status,objective_value,iterations,elapsed_seconds\n");
    for (int i = 0; i < count; i++) {
        printf("%s,%d,%d,%s,%.10f,%d,%.6f\n",
               results[i].filename, results[i].numVars, results[i].numConstraints,
               results[i].statusStr, results[i].objValue,
               results[i].iterations, results[i].elapsed);
    }
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
    
    lp->lowerBounds = (double*)calloc(2, sizeof(double));
    lp->upperBounds = (double*)malloc(2 * sizeof(double));
    lp->upperBounds[0] = DBL_MAX;
    lp->upperBounds[1] = DBL_MAX;
    lp->isInteger = (int*)calloc(2, sizeof(int));
    lp->rangeValues = (double*)calloc(2, sizeof(double));
    lp->objConstant = 0.0;
    
    return lp;
}

// ===========================================================================
// INTERACTIVE MODE
// ===========================================================================

/**
 * Solve a single file (shared logic for interactive + normal mode).
 * Returns the exit code (0 = OPTIMAL, 1 = other).
 */
int solveFile(const char* filename, cudaDeviceProp* prop) {
    LPProblem* lp = parseMPS(filename);
    if (!lp) return EXIT_FAILURE;
    
    preprocessBounds(lp);
    Tableau* tab = createTableau(lp);
    
    g_totalIterations = 0;
    double tstart = hpc_gettime();
    SimplexStatus status = solveSimplex(tab, lp);
    double elapsed = hpc_gettime() - tstart;
    
    outputSolution(tab, lp, status, elapsed);
    
    freeTableau(tab);
    freeLPProblem(lp);
    return (status == OPTIMAL) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/**
 * Interactive REPL.
 * Commands: help, quit/exit, set <option> <value>, or a filename to solve.
 */
void interactiveMode(cudaDeviceProp* prop) {
    char line[1024];
    
    printf("CUDA Simplex (Original) — Interactive Mode\n");
    printf("Type a .mps filename to solve, or 'help' for commands.\n\n");
    
    while (1) {
        printf("simplex> ");
        fflush(stdout);
        
        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;  // EOF
        }
        
        // Strip trailing newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        
        // Skip empty lines
        if (len == 0) continue;
        
        // Commands
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            break;
        } else if (strcmp(line, "help") == 0) {
            printf("Commands:\n");
            printf("  <file.mps>              Solve an MPS file\n");
            printf("  set verbose <0|1|2>     Set verbosity level\n");
            printf("  set debug <0|1>         Toggle tableau printing\n");
            printf("  set format <text|json|csv>  Output format\n");
            printf("  set maxiter <N>         Max iterations\n");
            printf("  set timeout <seconds>   Solve timeout (0=off)\n");
            printf("  status                  Show current settings\n");
            printf("  quit / exit             Exit\n");
        } else if (strcmp(line, "status") == 0) {
            printf("  verbose  = %d\n", g_verbose);
            printf("  debug    = %d\n", g_debug);
            printf("  format   = %s\n",
                   g_outputFormat == OUTPUT_JSON ? "json" :
                   g_outputFormat == OUTPUT_CSV  ? "csv"  : "text");
            printf("  maxiter  = %d\n", g_maxIter);
            printf("  timeout  = %.1f s\n", g_timeout);
            printf("  device   = %s\n", prop->name);
        } else if (strncmp(line, "set ", 4) == 0) {
            char key[64], val[64];
            if (sscanf(line + 4, "%63s %63s", key, val) == 2) {
                if (strcmp(key, "verbose") == 0) {
                    g_verbose = atoi(val);
                    printf("verbose = %d\n", g_verbose);
                } else if (strcmp(key, "debug") == 0) {
                    g_debug = atoi(val);
                    printf("debug = %d\n", g_debug);
                } else if (strcmp(key, "format") == 0) {
                    if (strcmp(val, "json") == 0)      g_outputFormat = OUTPUT_JSON;
                    else if (strcmp(val, "csv") == 0)  g_outputFormat = OUTPUT_CSV;
                    else                               g_outputFormat = OUTPUT_TEXT;
                    printf("format = %s\n", val);
                } else if (strcmp(key, "maxiter") == 0) {
                    g_maxIter = atoi(val);
                    printf("maxiter = %d\n", g_maxIter);
                } else if (strcmp(key, "timeout") == 0) {
                    g_timeout = atof(val);
                    printf("timeout = %.1f s\n", g_timeout);
                } else {
                    printf("Unknown option: %s\n", key);
                }
            } else {
                printf("Usage: set <option> <value>\n");
            }
        } else {
            // Treat as filename
            struct stat st;
            if (stat(line, &st) != 0) {
                printf("File not found: %s\n", line);
                continue;
            }
            solveFile(line, prop);
        }
    }
}

void printUsage(const char* progName) {
    fprintf(stderr, "CUDA Two-Phase Simplex Solver (Original Algorithm)\n\n");
    fprintf(stderr, "Usage: %s [options] <problem.mps> [...]\n\n", progName);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s, --silent         Suppress solver output\n");
    fprintf(stderr, "  -d, --debug          Print initial, intermediate, and final tableaux\n");
    fprintf(stderr, "  --diag               Enable diagnostic output (verbose=2)\n");
    fprintf(stderr, "  -i, --interactive    Interactive REPL mode\n");
    fprintf(stderr, "  -m, --max-iter <N>   Set maximum iterations (default: 50000)\n");
    fprintf(stderr, "  -t, --timeout <sec>  Set solve timeout in seconds (0=off)\n");
    fprintf(stderr, "  --json               Output solution in JSON format\n");
    fprintf(stderr, "  --csv                Output solution in CSV format\n");
    fprintf(stderr, "  --batch              Batch mode: solve multiple files, print summary\n");
    fprintf(stderr, "  --log <file>         Write per-iteration log to CSV file\n");
    fprintf(stderr, "  -h, --help           Show this help message\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s problem.mps\n", progName);
    fprintf(stderr, "  %s -d problem.mps              # show all tableaux\n", progName);
    fprintf(stderr, "  %s -m 1000 -t 5 problem.mps    # max 1000 iters, 5s timeout\n", progName);
    fprintf(stderr, "  %s -i                           # interactive mode\n", progName);
    fprintf(stderr, "  %s --json problem.mps\n", progName);
    fprintf(stderr, "  %s --batch netlib/*.mps\n", progName);
    fprintf(stderr, "  %s --batch netlib/\n", progName);
    fprintf(stderr, "  %s --log iter.csv problem.mps\n", progName);
}

int main(int argc, char* argv[]) {
    // Parse flags
    int batchMode = 0;
    int interactiveFlag = 0;
    const char* logFile = NULL;
    int inputCount = 0;
    const char** inputFiles = (const char**)malloc(argc * sizeof(const char*));
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--silent") == 0) {
            g_verbose = 0;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            g_debug = 1;
        } else if (strcmp(argv[i], "--diag") == 0) {
            g_verbose = 2;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactiveFlag = 1;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max-iter") == 0) {
            if (i + 1 < argc) {
                g_maxIter = atoi(argv[++i]);
                if (g_maxIter <= 0) { fprintf(stderr, "Error: --max-iter must be positive\n"); free(inputFiles); return EXIT_FAILURE; }
            } else {
                fprintf(stderr, "Error: --max-iter requires an integer argument\n");
                free(inputFiles); return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) {
            if (i + 1 < argc) {
                g_timeout = atof(argv[++i]);
                if (g_timeout < 0.0) { fprintf(stderr, "Error: --timeout must be non-negative\n"); free(inputFiles); return EXIT_FAILURE; }
            } else {
                fprintf(stderr, "Error: --timeout requires a numeric argument\n");
                free(inputFiles); return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--json") == 0) {
            g_outputFormat = OUTPUT_JSON;
        } else if (strcmp(argv[i], "--csv") == 0) {
            g_outputFormat = OUTPUT_CSV;
        } else if (strcmp(argv[i], "--batch") == 0) {
            batchMode = 1;
        } else if (strcmp(argv[i], "--log") == 0) {
            if (i + 1 < argc) {
                logFile = argv[++i];
            } else {
                fprintf(stderr, "Error: --log requires a filename argument\n");
                free(inputFiles);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            free(inputFiles);
            return EXIT_SUCCESS;
        } else if (argv[i][0] != '-') {
            inputFiles[inputCount++] = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            free(inputFiles);
            return EXIT_FAILURE;
        }
    }
    
    // In batch mode, auto-expand directory arguments into .mps files
    if (batchMode) {
        int expandedCount = 0;
        const char** expandedFiles = (const char**)malloc(4096 * sizeof(const char*));
        
        for (int i = 0; i < inputCount; i++) {
            struct stat st;
            if (stat(inputFiles[i], &st) == 0 && S_ISDIR(st.st_mode)) {
                DIR* dir = opendir(inputFiles[i]);
                if (dir) {
                    struct dirent* entry;
                    while ((entry = readdir(dir)) != NULL) {
                        int len = (int)strlen(entry->d_name);
                        if (len > 4 && strcmp(entry->d_name + len - 4, ".mps") == 0) {
                            char* fullpath = (char*)malloc(512);
                            snprintf(fullpath, 512, "%s/%s", inputFiles[i], entry->d_name);
                            expandedFiles[expandedCount++] = fullpath;
                        }
                    }
                    closedir(dir);
                }
            } else {
                expandedFiles[expandedCount++] = inputFiles[i];
            }
        }
        
        free(inputFiles);
        inputFiles = expandedFiles;
        inputCount = expandedCount;
    }
    
    // Open iteration log if requested
    if (logFile) {
        g_iterLog = fopen(logFile, "w");
        if (!g_iterLog) {
            fprintf(stderr, "Error: Cannot open log file %s\n", logFile);
            free((void*)inputFiles);
            return EXIT_FAILURE;
        }
        fprintf(g_iterLog, "iter,phase,pivot_col,pivot_row,reduced_cost,ratio,obj_rhs\n");
    }
    
    if (g_verbose && g_outputFormat == OUTPUT_TEXT) {
        printf("CUDA Two-Phase Simplex Solver (Original Algorithm)\n");
        printf("==================================================\n\n");
    }
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found!\n");
        if (g_iterLog) fclose(g_iterLog);
        free((void*)inputFiles);
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (g_verbose && g_outputFormat == OUTPUT_TEXT) {
        printf("Using CUDA device: %s\n", prop.name);
        printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    }
    
    int exitCode = EXIT_SUCCESS;
    
    // ===== INTERACTIVE MODE =====
    if (interactiveFlag) {
        interactiveMode(&prop);
        if (g_iterLog) fclose(g_iterLog);
        free((void*)inputFiles);
        return EXIT_SUCCESS;
    }
    
    if (batchMode && inputCount > 0) {
        // ===== BATCH MODE =====
        int savedVerbose = g_verbose;
        if (g_verbose < 2) g_verbose = 0;  // Silence per-problem output unless -d
        
        BatchResult* results = (BatchResult*)malloc(inputCount * sizeof(BatchResult));
        int resultCount = 0;
        
        for (int f = 0; f < inputCount; f++) {
            // Extract base filename for display
            const char* base = strrchr(inputFiles[f], '/');
            const char* displayName = base ? base + 1 : inputFiles[f];
            
            LPProblem* lp = parseMPS(inputFiles[f]);
            if (!lp) {
                snprintf(results[resultCount].filename, sizeof(results[resultCount].filename),
                         "%s", displayName);
                results[resultCount].numVars = 0;
                results[resultCount].numConstraints = 0;
                results[resultCount].statusStr = "PARSE_ERROR";
                results[resultCount].objValue = 0.0;
                results[resultCount].iterations = 0;
                results[resultCount].elapsed = 0.0;
                resultCount++;
                continue;
            }
            
            snprintf(results[resultCount].filename, sizeof(results[resultCount].filename),
                     "%s", displayName);
            results[resultCount].numVars = lp->numVars;
            results[resultCount].numConstraints = lp->numConstraints;
            
            preprocessBounds(lp);
            Tableau* tab = createTableau(lp);
            
            g_totalIterations = 0;
            
            double tstart = hpc_gettime();
            SimplexStatus status = solveSimplex(tab, lp);
            double elapsed = hpc_gettime() - tstart;
            
            results[resultCount].statusStr = statusString(status);
            results[resultCount].iterations = g_totalIterations;
            results[resultCount].elapsed = elapsed;
            
            if (status == OPTIMAL) {
                double* solution;
                results[resultCount].objValue = extractSolutionValues(tab, lp, &solution);
                free(solution);
            } else {
                results[resultCount].objValue = 0.0;
            }
            
            freeTableau(tab);
            freeLPProblem(lp);
            resultCount++;
            
            if (savedVerbose && g_outputFormat == OUTPUT_TEXT) {
                fprintf(stderr, "\rSolved %d/%d problems...", resultCount, inputCount);
                fflush(stderr);
            }
        }
        
        if (savedVerbose && g_outputFormat == OUTPUT_TEXT)
            fprintf(stderr, "\r                              \r");
        
        g_verbose = savedVerbose;
        
        // Print batch summary
        switch (g_outputFormat) {
            case OUTPUT_JSON:
                printBatchSummaryJSON(results, resultCount);
                break;
            case OUTPUT_CSV:
                printBatchSummaryCSV(results, resultCount);
                break;
            case OUTPUT_TEXT:
            default:
                printBatchSummaryText(results, resultCount);
                break;
        }
        
        free(results);
        
    } else {
        // ===== SINGLE FILE MODE =====
        LPProblem* lp = NULL;
        
        if (inputCount > 0) {
            if (g_verbose && g_outputFormat == OUTPUT_TEXT)
                printf("Loading problem from: %s\n\n", inputFiles[0]);
            lp = parseMPS(inputFiles[0]);
            if (!lp) {
                if (g_iterLog) fclose(g_iterLog);
                free((void*)inputFiles);
                return EXIT_FAILURE;
            }
        } else {
            if (g_outputFormat == OUTPUT_TEXT && g_verbose) {
                printf("No input file provided. Using test problem.\n\n");
                printf("Usage: %s [options] <problem.mps>\n\n", argv[0]);
            }
            lp = createTestProblem();
            
            if (g_verbose && g_outputFormat == OUTPUT_TEXT) {
                printf("Test Problem:\n");
                printf("  Maximize: 3*x1 + 2*x2\n");
                printf("  Subject to:\n");
                printf("    x1 + x2 <= 4\n");
                printf("    2*x1 + x2 <= 6\n");
                printf("  Expected: x1=2, x2=2, z=10\n");
            }
        }
        
        // Preprocess variable bounds and range constraints
        preprocessBounds(lp);
        
        // Create tableau
        Tableau* tab = createTableau(lp);
        
        g_totalIterations = 0;
        
        // Time only the computation (solving), not I/O
        double tstart = hpc_gettime();
        
        // Solve
        SimplexStatus status = solveSimplex(tab, lp);
        
        double tfinish = hpc_gettime();
        double elapsed = tfinish - tstart;
        
        // Output solution in the requested format
        outputSolution(tab, lp, status, elapsed);
        
        // Cleanup
        freeTableau(tab);
        freeLPProblem(lp);
        
        exitCode = (status == OPTIMAL) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    // Final cleanup
    if (g_iterLog) fclose(g_iterLog);
    free((void*)inputFiles);
    
    return exitCode;
}
