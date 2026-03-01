#ifndef SIMPLEX_IO_H
#define SIMPLEX_IO_H

#include "../core/simplex_core.h"

typedef struct {
    char filename[512];
    int numVars;
    int numConstraints;
    const char* statusStr;
    double objValue;
    int iterations;
    double elapsed;
} BatchResult;

void printSolution(Tableau* tab, LPProblem* lp, SimplexStatus status);
void printSolutionJSON(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed);
void printSolutionCSV(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed);
void outputSolution(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed);

void printBatchSummaryText(BatchResult* results, int count);
void printBatchSummaryJSON(BatchResult* results, int count);
void printBatchSummaryCSV(BatchResult* results, int count);

const char* statusString(SimplexStatus status);
double extractSolutionValues(Tableau* tab, LPProblem* lp, double** outSolution);

#endif
