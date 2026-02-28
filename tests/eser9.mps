* =========================================
* ESERCIZIO 9
* =========================================
* Minimize: -2*X1 + 5*X2 + 4*X3
* Subject to:
*   +4*X1 + X2 - 2*X3 <= +2
*   +2*X1 - 2*X2 + X3 <= +3
*   -X1 + X2 - 2*X3 <= -1
*   X1, X2, X3 >= 0
*
* NOTE: tests.md had X1 coeff in R3 = 2, corrected to -1 per image
NAME          ESER9
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 L  R3
COLUMNS
    X1        OBJ                 -2   R1                   4
    X1        R2                   2   R3                  -1
    X2        OBJ                  5   R1                   1
    X2        R2                  -2   R3                   1
    X3        OBJ                  4   R1                  -2
    X3        R2                   1   R3                  -2
RHS
    RHS1      R1                   2   R2                   3
    RHS1      R3                  -1
ENDATA
* Solution (Primal & Dual Simplex):
*   Soluzione ottima trovata!
*   Costo ottimo: -0.4000
*   X(1) = 0.6000
*   X(2) = 0.0000
*   X(3) = 0.2000
