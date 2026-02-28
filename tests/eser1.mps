* =========================================
* ESERCIZIO 1
* =========================================
* Minimize: -2*X1 - X2 + X3
* Subject to:
*   +X1 + X2 + 2*X3 <= 4
*   -X1 + 2*X2 + X3 <= 2
*   -X1 + X2 + X3 <= -2
*   X1, X2, X3 >= 0
NAME          ESER1
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 L  R3
COLUMNS
    X1        OBJ                 -2   R1                   1
    X1        R2                  -1   R3                  -1
    X2        OBJ                 -1   R1                   1
    X2        R2                   2   R3                   1
    X3        OBJ                  1   R1                   2
    X3        R2                   1   R3                   1
RHS
    RHS1      R1                   4   R2                   2
    RHS1      R3                  -2
ENDATA
* Solution: not provided in images
