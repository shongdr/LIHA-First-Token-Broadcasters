# qwen-instruct: joint-ablation plan

LCB screen qwen-instruct-lcbscreen (crosslingual, baseline 0.720); eligible heads 336 of 336 (FLORES screen qwen-instruct). Paper head L22H6: rank 2 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L16H9 | +0.400 | -0.007 |
| 2 | L22H6 | +0.220 | +0.228 |
| 3 | L21H2 | +0.160 | +0.007 |
| 4 | L1H6 | +0.130 | +0.005 |
| 5 | L15H4 | +0.114 | -0.002 |
| 6 | L15H11 | +0.074 | -0.010 |
| 7 | L21H3 | +0.070 | +0.011 |
| 8 | L11H8 | +0.070 | +0.018 |
| 9 | L14H8 | +0.053 | -0.000 |
| 10 | L2H1 | +0.053 | +0.001 |
| 11 | L0H1 | +0.053 | +0.067 |
| 12 | L23H8 | +0.050 | +0.001 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L16H9 | L16H8; L16H7; L16H3; L16H10; L16H4 |
| 2 | L16H9, L22H6 | L16H10, L22H2; L16H10, L22H4; L16H10, L22H5; L16H8, L22H5; L16H7, L22H2 |
| 4 | L16H9, L22H6, L21H2, L1H6 | L16H11, L22H0, L21H6, L1H9; L16H1, L22H4, L21H0, L1H3; L16H5, L22H2, L21H9, L1H2; L16H4, L22H0, L21H8, L1H8; L16H4, L22H10, L21H5, L1H1 |
