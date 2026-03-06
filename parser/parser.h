#ifndef SIMPLEX_PARSER_H
#define SIMPLEX_PARSER_H

#include "../core/simplex_core.h"

int hasFileExtensionIgnoreCase(const char* filename, const char* extension);
LPProblem* parseLP(const char* filename, const SolverConfig* config);
LPProblem* parseMPS(const char* filename, const SolverConfig* config);
void preprocessBounds(LPProblem* lp, const SolverConfig* config);
void freeLPProblem(LPProblem* lp);
LPProblem* createTestProblem();

#endif
