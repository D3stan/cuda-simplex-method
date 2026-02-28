* =========================================
* ESERCIZIO 11
* =========================================
* Minimize: +2*X1 + X2 - 2*X3
* Subject to:
*   +4*X1 + X2 - X3 <= +3
*   +X1 - 2*X2 + X3 <= +3
*   +2*X1 - X2 - X3 >= +1
*   X1, X2, X3 >= 0
*
* NOTE: tests.md had R3 as L type with RHS=3,
*       corrected to G type with RHS=1 per image
NAME          ESER11
OBJSENSE
 MIN
ROWS
 N  OBJ
 L  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                  2   R1                   4
    X1        R2                   1   R3                   2
    X2        OBJ                  1   R1                   1
    X2        R2                  -2   R3                  -1
    X3        OBJ                 -2   R1                  -1
    X3        R2                   1   R3                  -1
RHS
    RHS1      R1                   3   R2                   3
    RHS1      R3                   1
ENDATA
* Solution (Complementarity check):
*   x = (1, 0, 1) verificata OTTIMA
*   Costo ottimo: 0.0000
*   X(1) = 1.0000
*   X(2) = 0.0000
*   X(3) = 1.0000
