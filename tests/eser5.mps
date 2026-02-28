* =========================================
* ESERCIZIO 5
* =========================================
* Minimize: -X1 + X2 + 2*X3
* Subject to:
*   +X1 + 2*X2 - X3 <= +2
*   +2*X1 - X2 + X3 <= +3
*   +X1 - X2 + 2*X3 >= +3
*   X1, X2, X3 >= 0
NAME          ESER5
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                 -1   R1                   1
    X1        R2                   2   R3                   1
    X2        OBJ                  1   R1                   2
    X2        R2                  -1   R3                  -1
    X3        OBJ                  2   R1                  -1
    X3        R2                   1   R3                   2
RHS
    RHS1      R1                   2   R2                   3
    RHS1      R3                   3
ENDATA
* Solution (Primal & Dual Simplex):
*   Soluzione ottima trovata!
*   Costo ottimo: 1.0000
*   X(1) = 1.0000
*   X(2) = 0.0000
*   X(3) = 1.0000
