# olmo2-1b-instruct: joint-ablation plan

LCB screen olmo2-1b-instruct-lcbscreen (crosslingual, baseline 0.949); eligible heads 256 of 256 (FLORES screen olmo2-1b-instruct-screen). Paper head L12H8: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L12H8 | +0.343 | +0.136 |
| 2 | L1H4 | +0.081 | +0.082 |
| 3 | L2H6 | +0.063 | +0.894 |
| 4 | L8H10 | +0.061 | +0.009 |
| 5 | L0H10 | +0.053 | +0.160 |
| 6 | L9H9 | +0.051 | +0.060 |
| 7 | L5H5 | +0.040 | +0.002 |
| 8 | L5H6 | +0.040 | +0.003 |
| 9 | L12H4 | +0.040 | +0.009 |
| 10 | L9H15 | +0.040 | +0.011 |
| 11 | L15H2 | +0.040 | +0.039 |
| 12 | L12H2 | +0.030 | -0.013 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L12H8 | L12H4; L12H3; L12H13; L12H6; L12H14 |
| 2 | L12H8, L1H4 | L12H10, L1H12; L12H5, L1H10; L12H10, L1H15; L12H4, L1H15; L12H7, L1H3 |
| 4 | L12H8, L1H4, L2H6, L8H10 | L12H11, L1H0, L2H7, L8H15; L12H1, L1H14, L2H0, L8H14; L12H15, L1H12, L2H3, L8H13; L12H5, L1H14, L2H0, L8H3; L12H7, L1H14, L2H9, L8H11 |
