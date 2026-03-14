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
*
* =========================================
* SOLUZIONE ESERCIZIO 11
* =========================================
*
* d) Verifica ottimalita' di x = (1, 0, 1) con complementarieta'
* ---------------------------------------------------------------
*
* Verifica ammissibilita' di x = (1, 0, 1):
*   +4*x1 + x2 - x3 <= +3  =>  4 - 1 = 3       (saturo)
*   +x1 - 2*x2 + x3 <= +3  =>  1 + 1 = 2 < 3   (non saturo)
*   +2*x1 - x2 - x3 >= +1  =>  2 - 1 = 1        (saturo)
*
* Ammissibile (anche i vincoli di non negativita' sono rispettati).
*
* Duale di P:
*       max z = 3*w1 + 3*w2 + w3
*       s.t.  +4*w1 + w2 + 2*w3 <= +2
*             +w1 - 2*w2 - w3 <= +1
*             -w1 + w2 - w3 <= -2
*             w1, w2 <= 0, w3 >= 0
*
* Il secondo vincolo del primale non e' saturo => w2 = 0.
*
* Per gli scarti complementari, dato che x1 = 1 e x3 = 1,
* il primo e il terzo vincolo del duale devono essere saturi:
*   +4*w1 + 2*w3 = +2      (w2 = 0)
*   -w1 - w3 = -2          (w2 = 0)
*
*   Dal secondo: w1 = -w3 + 2  =>  w1 + w3 = 2
*   Sostituendo nel primo: 4*(-w3 + 2) + 2*w3 = 2  =>  WRONG, let me redo:
*
*   w1 + w3 = 2  (from -w1 - w3 = -2)
*   => w3 = 1 - 2*w1  (from 4*w1 + 2*w3 = 2  =>  2*w3 = 2 - 4*w1)
*   Sostituendo: w1 + (1 - 2*w1) = 2  =>  -w1 = 1  =>  w1 = -1
*   => w3 = 1 - 2*(-1) = 1 + 2 = 3
*
* Verifica vincoli di segno: w1 = -1 <= 0 OK, w2 = 0 <= 0 OK, w3 = 3 >= 0 OK
* Secondo vincolo duale: -1 - 0 - 3 = -4 <= 1 OK
*
* La soluzione duale rispetta i vincoli di segno e anche
* il secondo vincolo duale, quindi x = (1, 0, 1) e' ottima.
