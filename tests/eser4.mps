* =========================================
* ESERCIZIO 4
* =========================================
* Minimize: +5*X1 + 4*X2 - X3
* Subject to:
*   +2*X1 + X2 - 3*X3 <= +2
*   +4*X1 + 4*X2 + X3 <= +4
*   +2*X1 - X2 + X3 >= +1
*   X1, X2, X3 >= 0
*
* NOTE: tests.md had R3 RHS = 4, corrected to 1 per image/tableau
NAME          ESER4
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                  5   R1                   2
    X1        R2                   4   R3                   2
    X2        OBJ                  4   R1                   1
    X2        R2                   4   R3                  -1
    X3        OBJ                 -1   R1                  -3
    X3        R2                   1   R3                   1
RHS
    RHS1      R1                   2   R2                   4
    RHS1      R3                   1
ENDATA
* Solution (Primal & Dual Simplex):
*   Soluzione ottima trovata!
*   Costo ottimo: -4.0000
*   X(1) = 0.0000
*   X(2) = 0.0000
*   X(3) = 4.0000
