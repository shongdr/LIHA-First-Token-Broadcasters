# llama3.2-3b-instruct: joint-ablation plan

LCB screen llama3.2-3b-instruct-lcbscreen (crosslingual, baseline 0.939); eligible heads 672 of 672 (FLORES screen llama3.2-3b-instruct-screen). Paper head L13H19: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L13H19 | +0.323 | +0.008 |
| 2 | L8H13 | +0.061 | +0.003 |
| 3 | L0H0 | +0.041 | +0.008 |
| 4 | L12H1 | +0.040 | -0.005 |
| 5 | L10H13 | +0.040 | -0.003 |
| 6 | L13H8 | +0.040 | -0.000 |
| 7 | L5H14 | +0.040 | +0.004 |
| 8 | L12H23 | +0.040 | +0.017 |
| 9 | L12H7 | +0.039 | -0.001 |
| 10 | L11H0 | +0.030 | -0.012 |
| 11 | L12H3 | +0.030 | -0.002 |
| 12 | L11H11 | +0.030 | +0.000 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L13H19 | L13H9; L13H5; L13H2; L13H7; L13H20 |
| 2 | L13H19, L8H13 | L13H13, L8H17; L13H13, L8H21; L13H13, L8H23; L13H9, L8H23; L13H6, L8H18 |
| 4 | L13H19, L8H13, L0H0, L12H1 | L13H15, L8H0, L0H21, L12H9; L13H11, L8H21, L0H10, L12H2; L13H22, L8H18, L0H5, L12H17; L13H21, L8H0, L0H3, L12H6; L13H21, L8H12, L0H19, L12H12 |

Notes:
- k=4: the cut falls inside a tie on the screen (L12H1 and L10H13 both +0.040); the dNLL tie rule decides membership.
