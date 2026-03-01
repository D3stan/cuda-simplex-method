#ifndef SIMPLEX_SOLVER_H
#define SIMPLEX_SOLVER_H

#include "../core/simplex_core.h"

Tableau* createTableau(LPProblem* lp);
void freeTableau(Tableau* tab);
void syncTableauToHost(Tableau* tab);
void printTableau(Tableau* tab);
void printTableauStep(Tableau* tab, int iteration, int pivotRow, int pivotCol);
SimplexStatus solveSimplex(Tableau* tab, LPProblem* lp);

#endif
