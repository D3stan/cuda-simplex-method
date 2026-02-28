* =========================================
* ESERCIZIO 2
* =========================================
* Minimize: -2*X1 + 2*X2 + 2*X3
* Subject to:
*   +X1 + X2 - 2*X3 >= 1
*   +X1 + X2 - X3 <= 3
*   -X1 - X2 + 3*X3 >= -2
*   X1, X2, X3 >= 0
NAME          ESER2
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                 -2   R1                   1
    X1        R2                   1   R3                  -1
    X2        OBJ                  2   R1                   1
    X2        R2                   1   R3                  -1
    X3        OBJ                  2   R1                  -2
    X3        R2                  -1   R3                   3
RHS
    RHS1      R1                   1   R2                   3
    RHS1      R3                  -2
ENDATA
* Solution: not provided in images
