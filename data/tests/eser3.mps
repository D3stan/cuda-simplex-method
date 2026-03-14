* =========================================
* ESERCIZIO 3
* =========================================
* Minimize: -2*X1 + 2*X2 + 2*X3
* Subject to:
*   +2*X1 + X2 - X3 >= 1
*   +X1 + X2 + 3*X3 <= 4
*   -X1 - X2 + 3*X3 >= 1
*   X1, X2, X3 >= 0
NAME          ESER3
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  R1
 L  R2
 G  R3
COLUMNS
    X1        OBJ                 -2   R1                   2
    X1        R2                   1   R3                  -1
    X2        OBJ                  2   R1                   1
    X2        R2                   1   R3                  -1
    X3        OBJ                  2   R1                  -1
    X3        R2                   3   R3                   3
RHS
    RHS1      R1                   1   R2                   4
    RHS1      R3                   1
ENDATA
*
* =========================================
* SOLUZIONE ESERCIZIO 3
* =========================================
*
* a) Verifica ottimalita' di x = (4/5, 0, 3/5) con complementarieta'
* -------------------------------------------------------------------
*
* Verifica ammissibilita' di x = (4/5, 0, 3/5):
*   +2*x1 + x2 - x3 >= 1  =>  8/5 - 3/5 = 1         (saturo)
*   +x1 + x2 + 3*x3 <= 4  =>  4/5 + 9/5 = 13/5 < 4  (non saturo)
*   -x1 - x2 + 3*x3 >= 1  =>  -4/5 + 9/5 = 1         (saturo)
*
* Ammissibile (anche i vincoli di non negativita' sono rispettati).
*
* Duale di P:
*       max z = w1 + 4*w2 + w3
*       s.t.  +2*w1 + w2 - w3 <= -2
*             +w1 + w2 - w3 <= +2
*             -w1 + 3*w2 + 3*w3 <= +2
*             w1, w3 >= 0, w2 <= 0
*
* Il secondo vincolo del primale non e' saturo => w2 = 0.
*
* Per gli scarti complementari, dato che x1 = 4/5 e x3 = 3/5,
* il primo e il terzo vincolo del duale devono essere saturi:
*   +2*w1 - w3 = -2
*   -w1 + 3*w3 = 2
*
*   Dal primo: w3 = 2*w1 + 2
*   Sostituendo: -w1 + 3*(2*w1 + 2) = 2
*                -w1 + 6*w1 + 6 = 2
*                5*w1 = -4
*                w1 = -4/5
*   => w3 = 2*(-4/5) + 2 = -8/5 + 10/5 = 2/5
*
* Verifica vincoli di segno:
*   w1 = -4/5 < 0, ma dovrebbe essere w1 >= 0  =>  VIOLATO!
*
* Visto che la soluzione duale NON rispetta i vincoli di segno,
* allora la soluzione x = (4/5, 0, 3/5) NON e' ottima.
