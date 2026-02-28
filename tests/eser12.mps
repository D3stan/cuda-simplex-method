* =========================================
* ESERCIZIO 12
* =========================================
* Minimize: +2*X1 + X2 + 3*X3
* Subject to:
*   +2*X1 + X2 - 2*X3 <= +3
*   +X1 - 5*X2 + X3 >= +4
*   +2*X1 - 2*X2 - 2*X3 >= +3
*   X1, X2, X3 >= 0
NAME          ESER12
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 G  R2
 G  R3
COLUMNS
    X1        OBJ                  2   R1                   2
    X1        R2                   1   R3                   2
    X2        OBJ                  1   R1                   1
    X2        R2                  -5   R3                  -2
    X3        OBJ                  3   R1                  -2
    X3        R2                   1   R3                  -2
RHS
    RHS1      R1                   3   R2                   4
    RHS1      R3                   3
ENDATA
* Solution: not fully visible in images (cut off)
