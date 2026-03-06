#include "parser.h"

// ===========================================================================

/* Buffer and initial-capacity constants for the MPS parser */
#define MPS_LINE_BUF_SIZE   1024   /* max bytes per input line (incl. NUL) */
#define MPS_INIT_ROW_CAP      64   /* initial row array capacity           */
#define MPS_INIT_VAR_CAP      64   /* initial variable array capacity      */
#define MPS_INIT_COEFF_CAP   256   /* initial sparse-coefficient capacity  */

/* MPS fixed-column field boundaries (0-indexed, inclusive).
 * Reference: MPS format standard, fixed-column variant.
 *   Field 1 (type indicator): cols  1– 2
 *   Field 2 (name 1):         cols  4–11
 *   Field 3 (name 2):         cols 14–21
 *   Field 4 (value 1):        cols 24–35
 *   Field 5 (name 3):         cols 39–46
 *   Field 6 (value 2):        cols 49–60 */
#define MPS_F1_START  1
#define MPS_F1_END    2
#define MPS_F2_START  4
#define MPS_F2_END   11
#define MPS_F3_START 14
#define MPS_F3_END   21
#define MPS_F4_START 24
#define MPS_F4_END   35
#define MPS_F5_START 39
#define MPS_F5_END   46
#define MPS_F6_START 49
#define MPS_F6_END   60

int hasFileExtensionIgnoreCase(const char* filename, const char* extension) {
    if (!filename || !extension) return 0;
    size_t n = strlen(filename);
    size_t e = strlen(extension);
    if (n < e) return 0;

    const char* tail = filename + (n - e);
    for (size_t i = 0; i < e; i++) {
        char a = tail[i];
        char b = extension[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static LPProblem* parseDAT(const char* filename, const SolverConfig* config) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    int numVars = 0;
    int numRows = 0;
    if (fscanf(file, "%d %d", &numVars, &numRows) != 2 || numVars <= 0 || numRows <= 0) {
        fprintf(stderr, "Error: Invalid DAT header in %s (expected: <numVars> <numConstraints>)\n", filename);
        fclose(file);
        return NULL;
    }

    LPProblem* lp = (LPProblem*)calloc(1, sizeof(LPProblem));
    lp->numVars = numVars;
    lp->numConstraints = numRows;
    lp->sense = MINIMIZE;
    lp->objConstant = 0.0;

    const char* base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    strncpy(lp->name, base, sizeof(lp->name) - 1);
    lp->name[sizeof(lp->name) - 1] = '\0';

    lp->objCoeffs = (double*)calloc(numVars, sizeof(double));
    lp->rhs = (double*)malloc(numRows * sizeof(double));
    lp->constraintTypes = (ConstraintType*)malloc(numRows * sizeof(ConstraintType));
    lp->constraintMatrix = (double**)calloc(numRows, sizeof(double*));
    lp->lowerBounds = (double*)calloc(numVars, sizeof(double));
    lp->upperBounds = (double*)malloc(numVars * sizeof(double));
    lp->isInteger = (int*)calloc(numVars, sizeof(int));
    lp->rangeValues = (double*)calloc(numRows, sizeof(double));
    lp->varNames = (char**)calloc(numVars, sizeof(char*));
    lp->constraintNames = (char**)calloc(numRows, sizeof(char*));

    for (int j = 0; j < numVars; j++) lp->upperBounds[j] = DBL_MAX;

    for (int i = 0; i < numRows; i++) {
        lp->constraintMatrix[i] = (double*)calloc(numVars, sizeof(double));
    }

    for (int i = 0; i < numRows; i++) {
        if (fscanf(file, "%lf", &lp->rhs[i]) != 1) {
            fprintf(stderr, "Error: Invalid RHS data at constraint %d in %s\n", i + 1, filename);
            fclose(file);
            freeLPProblem(lp);
            return NULL;
        }
    }

    for (int i = 0; i < numRows; i++) {
        int typeFlag = 0;
        if (fscanf(file, "%d", &typeFlag) != 1) {
            fprintf(stderr, "Error: Invalid constraint type at row %d in %s\n", i + 1, filename);
            fclose(file);
            freeLPProblem(lp);
            return NULL;
        }

        if (typeFlag == 1) lp->constraintTypes[i] = CONSTRAINT_LE;
        else if (typeFlag == -1) lp->constraintTypes[i] = CONSTRAINT_GE;
        else if (typeFlag == 0) lp->constraintTypes[i] = CONSTRAINT_EQ;
        else {
            fprintf(stderr, "Warning: Unknown DAT constraint type %d at row %d; defaulting to EQ.\n",
                    typeFlag, i + 1);
            lp->constraintTypes[i] = CONSTRAINT_EQ;
        }
    }

    for (int j = 0; j < numVars; j++) {
        double objCoeff = 0.0;
        int nnz = 0;
        if (fscanf(file, "%lf %d", &objCoeff, &nnz) != 2 || nnz < 0) {
            fprintf(stderr, "Error: Invalid column header for variable %d in %s\n", j + 1, filename);
            fclose(file);
            freeLPProblem(lp);
            return NULL;
        }

        lp->objCoeffs[j] = objCoeff;

        for (int k = 0; k < nnz; k++) {
            int rowIndex = 0;
            double val = 0.0;
            if (fscanf(file, "%d %lf", &rowIndex, &val) != 2) {
                fprintf(stderr, "Error: Invalid sparse entry %d for variable %d in %s\n",
                        k + 1, j + 1, filename);
                fclose(file);
                freeLPProblem(lp);
                return NULL;
            }

            if (rowIndex < 1 || rowIndex > numRows) {
                fprintf(stderr, "Error: DAT row index out of range (%d) for variable %d in %s\n",
                        rowIndex, j + 1, filename);
                fclose(file);
                freeLPProblem(lp);
                return NULL;
            }

            lp->constraintMatrix[rowIndex - 1][j] = val;
        }
    }

    for (int j = 0; j < numVars; j++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "x%d", j + 1);
        lp->varNames[j] = strdup(buf);
    }

    for (int i = 0; i < numRows; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "c%d", i + 1);
        lp->constraintNames[i] = strdup(buf);
    }

    fclose(file);

    if (config && config->verbose) {
        printf("Parsed LP (DAT): %s\n", lp->name);
        printf("  Variables: %d\n", lp->numVars);
        printf("  Constraints: %d\n", lp->numConstraints);
        printf("  Sense: MINIMIZE\n");
    }

    return lp;
}

LPProblem* parseLP(const char* filename, const SolverConfig* config) {
    if (hasFileExtensionIgnoreCase(filename, ".dat")) {
        return parseDAT(filename, config);
    }
    return parseMPS(filename, config);
}

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

/* Identifies the current section while parsing an MPS file. */
typedef enum {
    SEC_NONE = 0,
    SEC_ROWS,
    SEC_COLUMNS,
    SEC_RHS,
    SEC_BOUNDS,
    SEC_RANGES,
    SEC_UNKNOWN   /* unrecognised section — data lines are silently skipped */
} MpsSection;

/* Internal parse name type */
typedef struct { char name[64]; } VarName;

/* Sparse coefficient entry accumulated during COLUMNS parsing */
typedef struct { int row; int col; double value; } CoeffEntry;

/**
 * All mutable state accumulated while parsing an MPS file.
 * Passed by pointer to every section-parsing helper so they share
 * the same dynamic arrays without C++ lambda captures.
 */
typedef struct {
    /* Row data */
    int           rowCap;
    int           numRows;
    char**        rowNames;
    ConstraintType* rowTypes;
    double*       rhsValues;
    double*       rangeVals;
    char*         objRowName;

    /* Variable data */
    int           varCap;
    int           numVars;
    VarName*      varNamesBuf;
    double*       objCoeffsTemp;
    double*       loBounds;
    double*       upBounds;
    int*          isInt;

    /* Sparse coefficients */
    int           coeffCap;
    int           numCoeffs;
    CoeffEntry*   coeffs;

    /* COLUMNS state */
    int           inIntegerBlock;
} MpsParseState;

/* ---------------------------------------------------------------------------
 * Dynamic-array growth helpers
 * Both return 1 on success, 0 if any realloc fails (leaving state unchanged).
 * --------------------------------------------------------------------------- */

static int mpsGrowVarArrays(MpsParseState* s) {
    int oldCap = s->varCap;
    int newCap = oldCap * 2;

    VarName* t1 = (VarName*)realloc(s->varNamesBuf,   newCap * sizeof(VarName));
    if (!t1) return 0;
    s->varNamesBuf = t1;

    double* t2 = (double*)realloc(s->objCoeffsTemp, newCap * sizeof(double));
    if (!t2) return 0;
    s->objCoeffsTemp = t2;

    double* t3 = (double*)realloc(s->loBounds, newCap * sizeof(double));
    if (!t3) return 0;
    s->loBounds = t3;

    double* t4 = (double*)realloc(s->upBounds, newCap * sizeof(double));
    if (!t4) return 0;
    s->upBounds = t4;

    int* t5 = (int*)realloc(s->isInt, newCap * sizeof(int));
    if (!t5) return 0;
    s->isInt = t5;

    for (int i = oldCap; i < newCap; i++) {
        s->objCoeffsTemp[i] = 0.0;
        s->loBounds[i]      = 0.0;
        s->upBounds[i]      = DBL_MAX;
        s->isInt[i]         = 0;
    }
    s->varCap = newCap;
    return 1;
}

static int mpsGrowRowArrays(MpsParseState* s) {
    int oldCap = s->rowCap;
    int newCap = oldCap * 2;

    char** t1 = (char**)realloc(s->rowNames, newCap * sizeof(char*));
    if (!t1) return 0;
    s->rowNames = t1;

    ConstraintType* t2 = (ConstraintType*)realloc(s->rowTypes, newCap * sizeof(ConstraintType));
    if (!t2) return 0;
    s->rowTypes = t2;

    double* t3 = (double*)realloc(s->rhsValues, newCap * sizeof(double));
    if (!t3) return 0;
    s->rhsValues = t3;

    double* t4 = (double*)realloc(s->rangeVals, newCap * sizeof(double));
    if (!t4) return 0;
    s->rangeVals = t4;

    for (int i = oldCap; i < newCap; i++) {
        s->rhsValues[i] = 0.0;
        s->rangeVals[i] = 0.0;
    }
    s->rowCap = newCap;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Name lookup / insertion helpers
 * --------------------------------------------------------------------------- */

/* Find variable index by name; returns -1 if not found. */
static int mpsFindVar(const MpsParseState* s, const char* name) {
    for (int i = 0; i < s->numVars; i++) {
        if (strcmp(s->varNamesBuf[i].name, name) == 0) return i;
    }
    return -1;
}

/* Add a new variable; returns its index, or -1 on allocation failure. */
static int mpsAddVar(MpsParseState* s, const char* name) {
    if (s->numVars >= s->varCap && !mpsGrowVarArrays(s)) return -1;
    strncpy(s->varNamesBuf[s->numVars].name, name, 63);
    s->varNamesBuf[s->numVars].name[63] = '\0';
    s->objCoeffsTemp[s->numVars] = 0.0;
    s->loBounds[s->numVars]      = 0.0;
    s->upBounds[s->numVars]      = DBL_MAX;
    s->isInt[s->numVars]         = s->inIntegerBlock ? 1 : 0;
    return s->numVars++;
}

/* Find row index; returns -1 if it names the objective row, -2 if not found. */
static int mpsFindRow(const MpsParseState* s, const char* name) {
    if (s->objRowName && strcmp(name, s->objRowName) == 0) return -1;
    for (int i = 0; i < s->numRows; i++) {
        if (strcmp(s->rowNames[i], name) == 0) return i;
    }
    return -2;
}

/* Append a sparse coefficient entry; returns 1 on success, 0 on realloc failure. */
static int mpsAddCoeff(MpsParseState* s, int row, int col, double val) {
    if (s->numCoeffs >= s->coeffCap) {
        int newCap = s->coeffCap * 2;
        CoeffEntry* tmp = (CoeffEntry*)realloc(s->coeffs, newCap * sizeof(CoeffEntry));
        if (!tmp) return 0;
        s->coeffs    = tmp;
        s->coeffCap  = newCap;
    }
    s->coeffs[s->numCoeffs].row   = row;
    s->coeffs[s->numCoeffs].col   = col;
    s->coeffs[s->numCoeffs].value = val;
    s->numCoeffs++;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Per-section data-line parsers
 * Each returns 1 to continue, 0 to signal a fatal allocation failure.
 * --------------------------------------------------------------------------- */

static int parseRowsLine(const char* line, MpsParseState* s) {
    char typeField[4], nameField[64];
    extractMPSField(line, MPS_F1_START, MPS_F1_END, typeField, sizeof(typeField));
    extractMPSField(line, MPS_F2_START, MPS_F2_END, nameField, sizeof(nameField));

    if (nameField[0] == '\0') return 1;

    char type = typeField[0];
    if (type == 'N') {
        /* Only the first N row is the objective (MPS convention). */
        if (!s->objRowName) {
            s->objRowName = strdup(nameField);
            if (!s->objRowName) return 0;
        } else {
            fprintf(stderr, "Warning: Multiple N rows found. "
                    "Using '%s' as objective, ignoring '%s'.\n",
                    s->objRowName, nameField);
        }
    } else {
        if (s->numRows >= s->rowCap && !mpsGrowRowArrays(s)) return 0;
        s->rowNames[s->numRows] = strdup(nameField);
        if (!s->rowNames[s->numRows]) return 0;
        switch (type) {
            case 'L': s->rowTypes[s->numRows] = CONSTRAINT_LE; break;
            case 'G': s->rowTypes[s->numRows] = CONSTRAINT_GE; break;
            case 'E': s->rowTypes[s->numRows] = CONSTRAINT_EQ; break;
            default:
                fprintf(stderr, "Warning: Unknown row type '%c' for '%s', "
                        "defaulting to EQ.\n", type, nameField);
                s->rowTypes[s->numRows] = CONSTRAINT_EQ;
        }
        s->numRows++;
    }
    return 1;
}

static int parseColumnsLine(const char* line, MpsParseState* s) {
    char field2[64], field3[64], field4[64], field5[64], field6[64];
    extractMPSField(line, MPS_F2_START, MPS_F2_END, field2, sizeof(field2));  /* column name  */
    extractMPSField(line, MPS_F3_START, MPS_F3_END, field3, sizeof(field3));  /* row name 1   */
    extractMPSField(line, MPS_F4_START, MPS_F4_END, field4, sizeof(field4));  /* value 1      */
    extractMPSField(line, MPS_F5_START, MPS_F5_END, field5, sizeof(field5));  /* row name 2   */
    extractMPSField(line, MPS_F6_START, MPS_F6_END, field6, sizeof(field6));  /* value 2      */

    if (field2[0] == '\0') return 1;

    /* Integer MARKER lines — update INTORG/INTEND state and skip. */
    if (strcmp(field3, "'MARKER'") == 0 || strcmp(field3, "MARKER") == 0) {
        if (strstr(field4, "INTORG") || strstr(field5, "INTORG") ||
            strstr(field4, "'INTORG'") || strstr(field5, "'INTORG'")) {
            s->inIntegerBlock = 1;
        } else if (strstr(field4, "INTEND") || strstr(field5, "INTEND") ||
                   strstr(field4, "'INTEND'") || strstr(field5, "'INTEND'")) {
            s->inIntegerBlock = 0;
        }
        return 1;
    }

    double val1;
    if (!parseMPSDouble(field4, &val1)) return 1;

    int varIdx = mpsFindVar(s, field2);
    if (varIdx < 0) {
        varIdx = mpsAddVar(s, field2);
        if (varIdx < 0) return 0;
    }

    /* First (row, value) pair */
    int rowIdx = mpsFindRow(s, field3);
    if (rowIdx == -1) {
        s->objCoeffsTemp[varIdx] = val1;
    } else if (rowIdx >= 0) {
        if (!mpsAddCoeff(s, rowIdx, varIdx, val1)) return 0;
    }

    /* Second (row, value) pair — optional */
    double val2;
    if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
        rowIdx = mpsFindRow(s, field5);
        if (rowIdx == -1) {
            s->objCoeffsTemp[varIdx] = val2;
        } else if (rowIdx >= 0) {
            if (!mpsAddCoeff(s, rowIdx, varIdx, val2)) return 0;
        }
    }
    return 1;
}

static void parseRhsLine(const char* line, MpsParseState* s, LPProblem* lp) {
    char field2[64], field3[64], field4[64], field5[64], field6[64];
    extractMPSField(line, MPS_F2_START, MPS_F2_END, field2, sizeof(field2));
    extractMPSField(line, MPS_F3_START, MPS_F3_END, field3, sizeof(field3));
    extractMPSField(line, MPS_F4_START, MPS_F4_END, field4, sizeof(field4));
    extractMPSField(line, MPS_F5_START, MPS_F5_END, field5, sizeof(field5));
    extractMPSField(line, MPS_F6_START, MPS_F6_END, field6, sizeof(field6));

    double val1;
    if (field3[0] != '\0' && parseMPSDouble(field4, &val1)) {
        int rowIdx = mpsFindRow(s, field3);
        if (rowIdx >= 0)       s->rhsValues[rowIdx] = val1;
        else if (rowIdx == -1) lp->objConstant = val1;  /* N-row RHS = objective constant */
    }

    double val2;
    if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
        int rowIdx = mpsFindRow(s, field5);
        if (rowIdx >= 0)       s->rhsValues[rowIdx] = val2;
        else if (rowIdx == -1) lp->objConstant = val2;
    }
}

static void parseBoundsLine(const char* line, MpsParseState* s) {
    /* Field 1 (cols 1-2):  bound type  (LO, UP, FX, FR, MI, PL, BV, LI, UI)
     * Field 2 (cols 4-11): bound set name (ignored — only one bound set supported)
     * Field 3 (cols 14-21): variable name
     * Field 4 (cols 24-35): value (absent for FR, MI, PL, BV) */
    char typeField[4], field2[64], field3[64], field4[64];
    extractMPSField(line, MPS_F1_START, MPS_F1_END, typeField, sizeof(typeField));
    extractMPSField(line, MPS_F2_START, MPS_F2_END, field2,    sizeof(field2));
    extractMPSField(line, MPS_F3_START, MPS_F3_END, field3,    sizeof(field3));
    extractMPSField(line, MPS_F4_START, MPS_F4_END, field4,    sizeof(field4));

    if (field3[0] == '\0') return;

    int varIdx = mpsFindVar(s, field3);
    if (varIdx < 0) {
        fprintf(stderr, "Warning: BOUNDS references unknown variable '%s'.\n", field3);
        return;
    }

    double val = 0.0;
    int hasValue = parseMPSDouble(field4, &val);

    if      (strcmp(typeField, "LO") == 0 && hasValue) { s->loBounds[varIdx] = val; }
    else if (strcmp(typeField, "UP") == 0 && hasValue) { s->upBounds[varIdx] = val; }
    else if (strcmp(typeField, "FX") == 0 && hasValue) { s->loBounds[varIdx] = val; s->upBounds[varIdx] = val; }
    else if (strcmp(typeField, "FR") == 0)             { s->loBounds[varIdx] = -DBL_MAX; s->upBounds[varIdx] = DBL_MAX; }
    else if (strcmp(typeField, "MI") == 0)             { s->loBounds[varIdx] = -DBL_MAX; }
    else if (strcmp(typeField, "PL") == 0)             { s->upBounds[varIdx] =  DBL_MAX; }
    else if (strcmp(typeField, "BV") == 0)             { s->loBounds[varIdx] = 0.0; s->upBounds[varIdx] = 1.0; s->isInt[varIdx] = 1; }
    else if (strcmp(typeField, "LI") == 0 && hasValue) { s->loBounds[varIdx] = val; s->isInt[varIdx] = 1; }
    else if (strcmp(typeField, "UI") == 0 && hasValue) { s->upBounds[varIdx] = val; s->isInt[varIdx] = 1; }
    else {
        fprintf(stderr, "Warning: Unknown/invalid bound type '%s' for variable '%s'.\n",
                typeField, field3);
    }
}

static void parseRangesLine(const char* line, MpsParseState* s) {
    /* Same field layout as RHS: set-name, row, value [, row, value] */
    char field2[64], field3[64], field4[64], field5[64], field6[64];
    extractMPSField(line, MPS_F2_START, MPS_F2_END, field2, sizeof(field2));
    extractMPSField(line, MPS_F3_START, MPS_F3_END, field3, sizeof(field3));
    extractMPSField(line, MPS_F4_START, MPS_F4_END, field4, sizeof(field4));
    extractMPSField(line, MPS_F5_START, MPS_F5_END, field5, sizeof(field5));
    extractMPSField(line, MPS_F6_START, MPS_F6_END, field6, sizeof(field6));

    double val1;
    if (field3[0] != '\0' && parseMPSDouble(field4, &val1)) {
        int rowIdx = mpsFindRow(s, field3);
        if (rowIdx >= 0) s->rangeVals[rowIdx] = val1;
    }

    double val2;
    if (field5[0] != '\0' && parseMPSDouble(field6, &val2)) {
        int rowIdx = mpsFindRow(s, field5);
        if (rowIdx >= 0) s->rangeVals[rowIdx] = val2;
    }
}

/* ---------------------------------------------------------------------------
 * Struct assembly: converts accumulated parse state into a fully allocated
 * LPProblem.  Returns 1 on success, 0 on allocation failure (lp partially
 * allocated; the caller should invoke freeLPProblem + freeParseState).
 * --------------------------------------------------------------------------- */

static int buildLPProblem(LPProblem* lp, MpsParseState* s) {
    int numVars = s->numVars;
    int numRows = s->numRows;

    lp->numVars        = numVars;
    lp->numConstraints = numRows;

    /* All pointer arrays use calloc so uninitialized entries are NULL,
     * keeping freeLPProblem safe when called from an error path. */
    lp->objCoeffs = (double*)malloc(numVars * sizeof(double));
    if (!lp->objCoeffs) return 0;
    memcpy(lp->objCoeffs, s->objCoeffsTemp, numVars * sizeof(double));

    lp->lowerBounds = (double*)malloc(numVars * sizeof(double));
    if (!lp->lowerBounds) return 0;
    memcpy(lp->lowerBounds, s->loBounds, numVars * sizeof(double));

    lp->upperBounds = (double*)malloc(numVars * sizeof(double));
    if (!lp->upperBounds) return 0;
    memcpy(lp->upperBounds, s->upBounds, numVars * sizeof(double));

    lp->isInteger = (int*)malloc(numVars * sizeof(int));
    if (!lp->isInteger) return 0;
    memcpy(lp->isInteger, s->isInt, numVars * sizeof(int));

    lp->rhs = (double*)malloc(numRows * sizeof(double));
    if (!lp->rhs) return 0;

    lp->constraintTypes = (ConstraintType*)malloc(numRows * sizeof(ConstraintType));
    if (!lp->constraintTypes) return 0;

    lp->constraintMatrix = (double**)calloc(numRows, sizeof(double*));
    if (!lp->constraintMatrix) return 0;

    lp->varNames = (char**)calloc(numVars, sizeof(char*));
    if (!lp->varNames) return 0;

    lp->constraintNames = (char**)calloc(numRows, sizeof(char*));
    if (!lp->constraintNames) return 0;

    lp->rangeValues = (double*)malloc(numRows * sizeof(double));
    if (!lp->rangeValues) return 0;

    for (int i = 0; i < numRows; i++) {
        lp->constraintMatrix[i] = (double*)calloc(numVars, sizeof(double));
        if (!lp->constraintMatrix[i]) return 0;
        lp->rhs[i]              = s->rhsValues[i];
        lp->constraintTypes[i]  = s->rowTypes[i];
        lp->constraintNames[i]  = s->rowNames[i];  /* Transfer ownership */
        lp->rangeValues[i]      = s->rangeVals[i];
    }

    for (int i = 0; i < numVars; i++) {
        lp->varNames[i] = strdup(s->varNamesBuf[i].name);
        if (!lp->varNames[i]) return 0;
    }

    /* Scatter sparse coefficients into the dense constraint matrix */
    for (int i = 0; i < s->numCoeffs; i++) {
        lp->constraintMatrix[s->coeffs[i].row][s->coeffs[i].col] = s->coeffs[i].value;
    }
    return 1;
}

/* Release all temporary buffers owned by the parse state.
 * Row-name pointers already transferred to lp->constraintNames are NOT freed
 * here; callers must pass the number of rows transferred so remaining ones
 * (not yet in lp) can still be freed. */
static void freeParseState(MpsParseState* s, int rowsTransferred) {
    free(s->rangeVals);
    free(s->rhsValues);
    free(s->coeffs);
    free(s->isInt);
    free(s->upBounds);
    free(s->loBounds);
    free(s->objCoeffsTemp);
    free(s->varNamesBuf);
    free(s->rowTypes);
    /* Free row-name strings not yet owned by lp->constraintNames */
    for (int i = rowsTransferred; i < s->numRows; i++) free(s->rowNames[i]);
    free(s->rowNames);
    free(s->objRowName);
}

/**
 * Parse an MPS file into an LPProblem.
 *
 * Single-pass, fixed-column format, dynamic allocation.
 * Responsibilities:
 *   1. File I/O  — open, read line-by-line, close
 *   2. Section dispatch  — route each data line to the appropriate parser
 *   3. Struct assembly  — delegate to buildLPProblem after the loop
 *
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
    lp->sense      = MINIMIZE;
    lp->objConstant = 0.0;

    char rawLine[MPS_LINE_BUF_SIZE];
    MpsSection section = SEC_NONE;

    /* Initialise parse state */
    MpsParseState s;
    memset(&s, 0, sizeof(s));
    s.rowCap       = MPS_INIT_ROW_CAP;
    s.varCap       = MPS_INIT_VAR_CAP;
    s.coeffCap     = MPS_INIT_COEFF_CAP;
    s.rowNames     = (char**)malloc(s.rowCap * sizeof(char*));
    s.rowTypes     = (ConstraintType*)malloc(s.rowCap * sizeof(ConstraintType));
    s.rhsValues    = (double*)calloc(s.rowCap, sizeof(double));
    s.rangeVals    = (double*)calloc(s.rowCap, sizeof(double));
    s.varNamesBuf  = (VarName*)malloc(s.varCap * sizeof(VarName));
    s.objCoeffsTemp = (double*)calloc(s.varCap, sizeof(double));
    s.loBounds     = (double*)calloc(s.varCap, sizeof(double));
    s.upBounds     = (double*)malloc(s.varCap * sizeof(double));
    s.isInt        = (int*)calloc(s.varCap, sizeof(int));
    s.coeffs       = (CoeffEntry*)malloc(s.coeffCap * sizeof(CoeffEntry));
    for (int i = 0; i < s.varCap; i++) s.upBounds[i] = DBL_MAX;

    /* ===================== MAIN PARSE LOOP ===================== */
    while (fgets(rawLine, MPS_LINE_BUF_SIZE, file)) {
        stripNewline(rawLine);
        if (rawLine[0] == '\0' || rawLine[0] == '*') continue;

        if (isMPSSectionHeader(rawLine)) {
            if (strncmp(rawLine, "NAME", 4) == 0) {
                if ((int)strlen(rawLine) > 14)
                    extractMPSField(rawLine, 14, 71, lp->name, sizeof(lp->name));
                continue;
            }
            if (strncmp(rawLine, "ROWS",    4) == 0) { section = SEC_ROWS;    continue; }
            if (strncmp(rawLine, "COLUMNS", 7) == 0) { section = SEC_COLUMNS; continue; }
            if (strncmp(rawLine, "RHS",     3) == 0) { section = SEC_RHS;     continue; }
            if (strncmp(rawLine, "BOUNDS",  6) == 0) { section = SEC_BOUNDS;  continue; }
            if (strncmp(rawLine, "RANGES",  6) == 0) { section = SEC_RANGES;  continue; }
            if (strncmp(rawLine, "ENDATA",  6) == 0) break;
            if (strncmp(rawLine, "OBJSENSE", 8) == 0) {
                if (fgets(rawLine, MPS_LINE_BUF_SIZE, file)) {
                    stripNewline(rawLine);
                    char senseBuf[16];
                    extractMPSField(rawLine, 0, 15, senseBuf, sizeof(senseBuf));
                    char* s2 = senseBuf;
                    while (*s2 == ' ' || *s2 == '\t') s2++;
                    if      (strncmp(s2, "MAX", 3) == 0) lp->sense = MAXIMIZE;
                    else if (strncmp(s2, "MIN", 3) == 0) lp->sense = MINIMIZE;
                }
                continue;
            }
            section = SEC_UNKNOWN;
            continue;
        }

        /* Data lines — dispatch by current section */
        switch (section) {
            case SEC_ROWS:
                if (!parseRowsLine(rawLine, &s)) goto cleanup;
                break;
            case SEC_COLUMNS:
                if (!parseColumnsLine(rawLine, &s)) goto cleanup;
                break;
            case SEC_RHS:
                parseRhsLine(rawLine, &s, lp);
                break;
            case SEC_BOUNDS:
                parseBoundsLine(rawLine, &s);
                break;
            case SEC_RANGES:
                parseRangesLine(rawLine, &s);
                break;
            default:
                break;  /* SEC_UNKNOWN: silently skip */
        }
    }

    fclose(file);
    file = NULL;

    /* ===================== BUILD LPProblem STRUCTURE ===================== */
    if (!buildLPProblem(lp, &s)) goto cleanup;

    /* Free temporary storage (row names transferred to lp->constraintNames) */
    freeParseState(&s, s.numRows);

    if (config && config->verbose) {
        printf("Parsed LP: %s\n", lp->name);
        printf("  Variables: %d\n", lp->numVars);
        printf("  Constraints: %d\n", lp->numConstraints);
        printf("  Sense: %s\n", lp->sense == MAXIMIZE ? "MAXIMIZE" : "MINIMIZE");
        if (lp->objConstant != 0.0)
            printf("  Objective constant: %.6f\n", lp->objConstant);
    }

    return lp;

cleanup:
    if (file) fclose(file);
    /* Free row names not yet transferred (lp->numConstraints tracks how many were) */
    freeParseState(&s, lp->numConstraints);
    freeLPProblem(lp);
    return NULL;
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

