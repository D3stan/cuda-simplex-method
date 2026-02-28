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
*
* =========================================
* SOLUZIONE ESERCIZIO 6
* =========================================
*
* a) Verifica ottimalita' di x = (4/5, 3/5, 1) con complementarieta'
* -------------------------------------------------------------------
*
* Verifica ammissibilita' di x = (4/5, 3/5, 1):
*   +x1 + 2*x2 - x3 <= +1  =>  4/5 + 6/5 - 1 = 1    (saturo)
*   +2*x1 - x2 + x3 <= +2  =>  8/5 - 3/5 + 1 = 2    (saturo)
*   +2*x1 - x2 + 2*x3 >= +3 =>  8/5 - 3/5 + 2 = 3   (saturo)
*
* Ammissibile (anche i vincoli di non negativita' sono rispettati).
*
* Duale di P:
*       max z = w1 + 2*w2 + 3*w3
*       s.t.  +w1 + 2*w2 + 2*w3 <= -1
*             +2*w1 - w2 - w3 <= +1
*             -w1 + w2 + 2*w3 <= +2
*             w1, w2 <= 0, w3 >= 0
*
* I vincoli del primale sono tutti saturi, quindi per gli scarti
* complementari le variabili duali possono essere anche non nulle.
*
* Dato che x1 = 4/5 > 0, x2 = 3/5 > 0 e x3 = 1 > 0, allora tutti
* i vincoli del duale devono essere saturi:
*   +w1 + 2*w2 + 2*w3 = -1
*   +2*w1 - w2 - w3 = +1
*   -w1 + w2 + 2*w3 = +2
*
* Risoluzione per eliminazione di Gauss:
*
*   [1   2   2 | -1]     [1   2   2 | -1]
*   [2  -1  -1 |  1]  => [0  -5  -5 |  3]     (R2 = R2 - 2*R1)
*   [-1  1   2 |  2]     [0   3   4 |  1]     (R3 = R3 + R1)
*
*   [1   0   0 |  1/5]
*   [0   1   1 | -3/5]    (R2 /= -5, R1 = R1 - 2*R2)
*   [0   0   1 | 14/5]    (R3 = R3 - 3*R2)
*
*   => w1 = 1/5, w2 = -3/5 - 14/5 = -17/5, w3 = 14/5
*
* Verifica vincoli di segno:
*   w1 = 1/5 > 0, ma dovrebbe essere w1 <= 0  =>  VIOLATO!
*
* Visto che la soluzione duale NON rispetta i vincoli di segno,
* allora la soluzione x = (4/5, 3/5, 1) NON e' ottima.
