# qwen2.5-7b-instruct: joint-ablation plan

LCB screen qwen2.5-7b-instruct-lcbscreen (crosslingual, baseline 0.960); eligible heads 780 of 784 (FLORES screen qwen2.5-7b-instruct-screen). Paper head L19H1: rank 4 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L0H1 | +0.232 | +0.094 |
| 2 | L0H26 | +0.190 | +0.227 |
| 3 | L0H3 | +0.042 | +0.554 |
| 4 | L19H1 | +0.040 | +0.026 |
| 5 | L19H0 | +0.040 | +0.036 |
| 6 | L3H6 | +0.030 | +0.004 |
| 7 | L5H1 | +0.020 | -0.002 |
| 8 | L1H16 | +0.020 | -0.002 |
| 9 | L2H22 | +0.020 | -0.001 |
| 10 | L20H26 | +0.020 | -0.001 |
| 11 | L9H0 | +0.020 | -0.001 |
| 12 | L10H10 | +0.020 | -0.000 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L0H1 | L0H18; L0H7; L0H2; L0H9; L0H20 |
| 2 | L0H1, L0H26 | L0H14, L0H18; L0H14, L0H27; L0H14, L0H5; L0H4, L0H7; L0H8, L0H18 |
| 4 | L0H1, L0H26, L0H3, L19H1 | L0H16, L0H0, L0H4, L19H5; L0H12, L0H27, L0H10, L19H20; L0H27, L0H18, L0H9, L19H18; L0H21, L0H0, L0H7, L19H27; L0H21, L0H13, L0H2, L19H13 |

Notes:
- k=4: the cut falls inside a tie on the screen (L19H1 and L19H0 both +0.040); the dNLL tie rule decides membership.
