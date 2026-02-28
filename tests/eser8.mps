* =========================================
* ESERCIZIO 8
* =========================================
* Minimize: -2*X1 - 2*X2 + 2*X3
* Subject to:
*   +X1 - X2 - 2*X3 >= +1
*   -X1 + 2*X2 + X3 <= +3
*   +2*X1 - 2*X2 + 2*X3 <= +4
*   X1, X2, X3 >= 0
NAME          ESER8
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  R1
 L  R2
 L  R3
COLUMNS
    X1        OBJ                 -2   R1                   1
    X1        R2                  -1   R3                   2
    X2        OBJ                 -2   R1                  -1
    X2        R2                   2   R3                  -2
    X3        OBJ                  2   R1                  -2
    X3        R2                   1   R3                   2
RHS
    RHS1      R1                   1   R2                   3
    RHS1      R3                   4
ENDATA
* Solution (Complementarity check):
*   x = (5, 4, 0) verificata NON ottima
*   (La soluzione duale NON rispetta i vincoli di segno)
