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
*
* =========================================
* SOLUZIONE ESERCIZIO 8
* =========================================
*
* a) Verifica ottimalita' di x = (5, 4, 0) con complementarieta'
* ---------------------------------------------------------------
*
* Verifica ammissibilita' di x = (5, 4, 0):
*   +x1 - x2 - 2*x3 >= +1  =>  5 - 4 = 1       (saturo)
*   -x1 + 2*x2 + x3 <= +3  =>  -5 + 8 = 3      (saturo)
*   +2*x1 - 2*x2 + 2*x3 <= +4  =>  10 - 8 = 2 < 4  (non saturo)
*
* Ammissibile (anche i vincoli di non negativita' sono rispettati).
*
* Duale di P:
*       max z = w1 + 3*w2 + 4*w3
*       s.t.  +w1 - w2 + 2*w3 <= -2
*             -w1 + 2*w2 - 2*w3 <= -2
*             -2*w1 + w2 + 2*w3 <= +2
*             w2, w3 <= 0, w1 >= 0
*
* Il terzo vincolo del primale non e' saturo => w3 = 0.
*
* Per gli scarti complementari, dato che x1 = 5 e x2 = 4,
* il primo e il secondo vincolo del duale devono essere saturi:
*   +w1 - w2 + 2*w3 = -2   =>  +w1 - w2 = -2    (w3 = 0)
*   -w1 + 2*w2 - 2*w3 = -2 =>  -w1 + 2*w2 = -2  (w3 = 0)
*
*   Da: w1 = w2 - 2
*   Sostituendo: -(w2 - 2) + 2*w2 = -2
*                -w2 + 2 + 2*w2 = -2
*                w2 = -4
*   => w1 = -4 - 2 = -6
*
* Verifica vincoli di segno:
*   w1 = -6 < 0, ma dovrebbe essere w1 >= 0  =>  VIOLATO!
*
* Visto che la soluzione duale NON rispetta i vincoli di segno,
* allora la soluzione x = (5, 4, 0) NON e' ottima.
