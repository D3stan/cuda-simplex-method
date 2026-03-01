#include "io.h"
#include "../solver/solver.h"

// ===========================================================================

/**
 * Helper: extract solution values and compute objective
 */
double extractSolutionValues(Tableau* tab, LPProblem* lp, double** outSolution) {
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

const char* statusString(SimplexStatus status) {
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

