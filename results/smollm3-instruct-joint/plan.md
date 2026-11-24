# smollm3-instruct: joint-ablation plan

LCB screen smollm3-instruct-lcbscreen (crosslingual, baseline 0.939); eligible heads 575 of 576 (FLORES screen smollm3-instruct-screen). Paper head None: rank None among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L15H4 | +0.031 | -0.005 |
| 2 | L21H7 | +0.031 | +0.022 |
| 3 | L34H8 | +0.030 | -0.138 |
| 4 | L17H6 | +0.030 | -0.001 |
| 5 | L1H13 | +0.020 | -0.032 |
| 6 | L13H6 | +0.020 | -0.027 |
| 7 | L30H2 | +0.020 | -0.001 |
| 8 | L16H15 | +0.020 | +0.002 |
| 9 | L15H14 | +0.020 | +0.003 |
| 10 | L19H0 | +0.020 | +0.005 |
| 11 | L20H3 | +0.020 | +0.015 |
| 12 | L16H4 | +0.020 | -0.007 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L15H4 | L15H5; L15H3; L15H13; L15H7; L15H14 |
| 2 | L15H4, L21H7 | L15H10, L21H12; L15H6, L21H10; L15H10, L21H15; L15H5, L21H15; L15H8, L21H3 |
| 4 | L15H4, L21H7, L34H8, L17H6 | L15H11, L21H0, L34H6, L17H14; L15H1, L21H14, L34H0, L17H13; L15H15, L21H12, L34H3, L17H12; L15H6, L21H14, L34H0, L17H2; L15H8, L21H14, L34H9, L17H10 |

Notes:
- k=1: the cut falls inside a tie on the screen (L15H4 and L21H7 both +0.031); the dNLL tie rule decides membership.
