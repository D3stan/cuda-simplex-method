* =========================================
* ESERCIZIO 6
* =========================================
* Minimize: -X1 + X2 + 2*X3
* Subject to:
*   +X1 + 2*X2 - X3 <= +1
*   +2*X1 - X2 + X3 <= +2
*   +2*X1 - X2 + 2*X3 >= +3
*   X1, X2, X3 >= 0
NAME          ESER6
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                 -1   R1                   1
    X1        R2                   2   R3                   2
    X2        OBJ                  1   R1                   2
    X2        R2                  -1   R3                  -1
    X3        OBJ                  2   R1                  -1
    X3        R2                   1   R3                   2
RHS
    RHS1      R1                   1   R2                   2
    RHS1      R3                   3
ENDATA
* Solution (Complementarity check):
*   x = (4/5, 3/5, 1) verificata NON ottima
*   (La soluzione duale NON rispetta i vincoli di segno)
