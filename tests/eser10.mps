* =========================================
* ESERCIZIO 10
* =========================================
* Minimize: +2*X1 + 5*X2 + 4*X3
* Subject to:
*   +2*X1 + 3*X2 - X3 <= -3
*   -2*X1 + 2*X2 + 5*X3 <= +5
*   -X1 + X2 - 2*X3 >= -1
*   X1, X2, X3 >= 0
NAME          ESER10
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                  2   R1                   2
    X1        R2                  -2   R3                  -1
    X2        OBJ                  5   R1                   3
    X2        R2                   2   R3                   1
    X3        OBJ                  4   R1                  -1
    X3        R2                   5   R3                  -2
RHS
    RHS1      R1                  -3   R2                   5
    RHS1      R3                  -1
ENDATA
* Solution (Primal & Dual Simplex):
*   Non esiste soluzione!
*   (La Fase 2 Termina con la funzione obiettivo positiva)
