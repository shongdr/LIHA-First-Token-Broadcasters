# olmo3-7b-instruct: joint-ablation plan

LCB screen olmo3-7b-instruct-lcbscreen (crosslingual, baseline 0.898); eligible heads 1024 of 1024 (FLORES screen olmo3-7b-instruct-screen). Paper head L14H25: rank 7 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L17H5 | +0.060 | +0.014 |
| 2 | L28H22 | +0.060 | +0.038 |
| 3 | L21H24 | +0.031 | -0.001 |
| 4 | L1H17 | +0.031 | +0.001 |
| 5 | L9H17 | +0.029 | +0.000 |
| 6 | L12H25 | +0.029 | +0.001 |
| 7 | L14H25 | +0.023 | +0.202 |
| 8 | L30H19 | +0.022 | +0.013 |
| 9 | L14H30 | +0.020 | -0.003 |
| 10 | L7H17 | +0.020 | -0.001 |
| 11 | L3H25 | +0.020 | -0.001 |
| 12 | L8H17 | +0.020 | -0.000 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L17H5 | L17H9; L17H25; L17H19; L17H27; L17H2 |
| 2 | L17H5, L28H22 | L17H13, L28H16; L17H29, L28H13; L17H13, L28H23; L17H28, L28H23; L17H6, L28H27 |
| 4 | L17H5, L28H22, L21H24, L1H17 | L17H15, L28H0, L21H4, L1H21; L17H11, L28H20, L21H1, L1H2; L17H21, L28H17, L21H26, L1H16; L17H29, L28H20, L21H0, L1H24; L17H4, L28H20, L21H8, L1H12 |

Notes:
- k=1: the cut falls inside a tie on the screen (L17H5 and L28H22 both +0.060); the dNLL tie rule decides membership.
