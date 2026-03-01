#ifndef SIMPLEX_PARSER_H
#define SIMPLEX_PARSER_H

#include "../core/simplex_core.h"

LPProblem* parseMPS(const char* filename);
void preprocessBounds(LPProblem* lp);
void freeLPProblem(LPProblem* lp);
LPProblem* createTestProblem();

#endif
