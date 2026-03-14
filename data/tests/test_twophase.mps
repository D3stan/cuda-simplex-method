NAME          Phase2Test
* Test problem requiring Phase 1 (has equality and >= constraints)
* 
* Minimize: 2*x1 + 3*x2 + x3
* Subject to:
*   x1 + x2 + x3 = 10     (equality - needs artificial)
*   x1 + 2*x2     >= 6    (>= constraint - needs surplus + artificial)
*       x2 + x3  <= 8     (<= constraint - needs slack only)
*   x1, x2, x3 >= 0
*
* This requires the two-phase method because of artificial variables
OBJSENSE
 MIN
ROWS
 N  COST
 E  EQ1
 G  GE1
 L  LE1
COLUMNS
    X1        COST                 2   EQ1                  1
    X1        GE1                  1
    X2        COST                 3   EQ1                  1
    X2        GE1                  2   LE1                  1
    X3        COST                 1   EQ1                  1
    X3        LE1                  1
RHS
    RHS1      EQ1                 10
    RHS1      GE1                  6
    RHS1      LE1                  8
ENDATA
