# gemma3-1b-instruct: joint-ablation plan

LCB screen gemma3-1b-instruct-lcbscreen-mono (monolingual, baseline 0.980); eligible heads 104 of 104 (FLORES screen gemma3-1b-instruct-screen). Paper head L11H3: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L11H3 | +0.660 | +0.720 |
| 2 | L17H1 | +0.030 | +0.061 |
| 3 | L4H0 | +0.020 | +0.071 |
| 4 | L24H0 | +0.010 | -0.026 |
| 5 | L22H3 | +0.010 | +0.008 |
| 6 | L20H1 | +0.010 | +0.017 |
| 7 | L20H3 | +0.010 | +0.023 |
| 8 | L3H0 | +0.010 | +0.098 |
| 9 | L0H0 | +0.010 | +0.115 |
| 10 | L15H1 | +0.010 | +0.126 |
| 11 | L16H0 | +0.010 | +0.240 |
| 12 | L10H1 | +0.000 | -0.134 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L11H3 | L11H2; L11H0; L11H1 |
| 2 | L11H3, L17H1 | L11H0, L17H2; L11H2, L17H0; L11H1, L17H2; L11H2, L17H2; L11H1, L17H3 |
| 4 | L11H3, L17H1, L4H0, L24H0 | L11H0, L17H0, L4H2, L24H3; L11H0, L17H2, L4H1, L24H2; L11H1, L17H2, L4H3, L24H2; L11H2, L17H2, L4H1, L24H3; L11H1, L17H0, L4H2, L24H1 |

Notes:
- k=1: only 3 distinct random sets exist under the layer rule.
- k=4: the cut falls inside a tie on the screen (L24H0 and L22H3 both +0.010); the dNLL tie rule decides membership.
