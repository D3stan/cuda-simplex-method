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
void printSolutionJSON(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed, const RunContext* run);
void printSolutionCSV(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed, const RunContext* run);
void outputSolution(Tableau* tab, LPProblem* lp, SimplexStatus status, double elapsed, const SolverConfig* config, const RunContext* run);

void printBatchSummaryText(BatchResult* results, int count);
void printBatchSummaryJSON(BatchResult* results, int count);
void printBatchSummaryCSV(BatchResult* results, int count);

const char* statusString(SimplexStatus status);
double extractSolutionValues(Tableau* tab, LPProblem* lp, double** outSolution);

#endif
