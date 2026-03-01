#include "parser.h"

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
LPProblem* parseMPS(const char* filename, const SolverConfig* config) {
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
    
    if (config && config->verbose) {
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
void preprocessBounds(LPProblem* lp, const SolverConfig* config) {
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
    
    if (config && config->verbose) {
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

