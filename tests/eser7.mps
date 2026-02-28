* =========================================
* ESERCIZIO 7
* =========================================
* Minimize: -2*X1 + X2 - 4*X3
* Subject to:
*   +X1 + 2*X2 - 3*X3 >= +2
*   +2*X1 - X2 + X3 <= -3
*   +4*X1 - 5*X2 + 3*X3 <= +3
*   X1, X2, X3 >= 0
NAME          ESER7
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  R1
 L  R2
 L  R3
COLUMNS
    X1        OBJ                 -2   R1                   1
    X1        R2                   2   R3                   4
    X2        OBJ                  1   R1                   2
    X2        R2                  -1   R3                  -5
    X3        OBJ                 -4   R1                  -3
    X3        R2                   1   R3                   3
RHS
    RHS1      R1                   2   R2                  -3
    RHS1      R3                   3
ENDATA
* Solution (Primal & Dual Simplex):
*   Soluzione Illimitata!
