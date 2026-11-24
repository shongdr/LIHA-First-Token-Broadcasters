# qwen3-4b-instruct: joint-ablation plan

LCB screen qwen3-4b-instruct-lcbscreen (crosslingual, baseline 0.908); eligible heads 1152 of 1152 (FLORES screen qwen3-4b-instruct-screen). Paper head L24H27: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L24H27 | +0.102 | +0.065 |
| 2 | L18H9 | +0.020 | -0.002 |
| 3 | L17H15 | +0.011 | +0.000 |
| 4 | L7H10 | +0.011 | +0.000 |
| 5 | L10H6 | +0.011 | +0.001 |
| 6 | L9H17 | +0.011 | +0.001 |
| 7 | L17H4 | +0.010 | -0.005 |
| 8 | L18H24 | +0.010 | -0.001 |
| 9 | L17H27 | +0.010 | -0.000 |
| 10 | L10H19 | +0.010 | -0.000 |
| 11 | L26H29 | +0.010 | -0.000 |
| 12 | L10H2 | +0.010 | +0.000 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L24H27 | L24H9; L24H25; L24H19; L24H28; L24H2 |
| 2 | L24H27, L18H9 | L24H13, L18H16; L24H3, L18H13; L24H13, L18H22; L24H29, L18H22; L24H6, L18H27 |
| 4 | L24H27, L18H9, L17H15, L7H10 | L24H15, L18H0, L17H5, L7H21; L24H11, L18H20, L17H1, L7H2; L24H21, L18H17, L17H26, L7H17; L24H3, L18H20, L17H0, L7H24; L24H5, L18H20, L17H9, L7H13 |

Notes:
- k=4: the cut falls inside a tie on the screen (L7H10 and L10H6 both +0.011); the dNLL tie rule decides membership.
