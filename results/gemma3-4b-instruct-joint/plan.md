# gemma3-4b-instruct: joint-ablation plan

LCB screen gemma3-4b-instruct-lcbscreen-mono (monolingual, baseline 1.000); eligible heads 272 of 272 (FLORES screen gemma3-4b-instruct-screen). Paper head L24H0: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L24H0 | +0.340 | +0.202 |
| 2 | L20H7 | +0.030 | -0.107 |
| 3 | L33H7 | +0.010 | -0.136 |
| 4 | L15H6 | +0.010 | -0.045 |
| 5 | L14H4 | +0.010 | -0.035 |
| 6 | L14H1 | +0.010 | -0.020 |
| 7 | L2H4 | +0.010 | -0.011 |
| 8 | L9H4 | +0.010 | -0.008 |
| 9 | L5H7 | +0.010 | -0.007 |
| 10 | L11H6 | +0.010 | -0.005 |
| 11 | L7H6 | +0.010 | -0.005 |
| 12 | L32H7 | +0.010 | -0.001 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L24H0 | L24H6; L24H5; L24H3; L24H7; L24H4 |
| 2 | L24H0, L20H7 | L24H2, L20H2; L24H6, L20H1; L24H2, L20H3; L24H6, L20H3; L24H7, L20H4 |
| 4 | L24H0, L20H7, L33H7, L15H6 | L24H2, L20H0, L33H6, L15H3; L24H1, L20H3, L33H0, L15H2; L24H4, L20H2, L33H4, L15H2; L24H6, L20H3, L33H0, L15H4; L24H7, L20H3, L33H1, L15H2 |

Notes:
- k=4: the cut falls inside a tie on the screen (L15H6 and L14H4 both +0.010); the dNLL tie rule decides membership.
