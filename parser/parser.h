#ifndef SIMPLEX_PARSER_H
#define SIMPLEX_PARSER_H

#include "../core/simplex_core.h"

LPProblem* parseMPS(const char* filename, const SolverConfig* config);
void preprocessBounds(LPProblem* lp, const SolverConfig* config);
void freeLPProblem(LPProblem* lp);
LPProblem* createTestProblem();

#endif
