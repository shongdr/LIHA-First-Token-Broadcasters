# qwen3-1.7b-instruct: joint-ablation plan

LCB screen qwen3-1.7b-instruct-lcbscreen (crosslingual, baseline 0.838); eligible heads 446 of 448 (FLORES screen qwen3-1.7b-instruct-screen). Paper head L18H12: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L18H12 | +0.485 | +0.128 |
| 2 | L6H0 | +0.051 | +0.009 |
| 3 | L9H3 | +0.048 | +0.001 |
| 4 | L18H14 | +0.032 | +0.000 |
| 5 | L12H3 | +0.030 | -0.001 |
| 6 | L5H7 | +0.030 | +0.005 |
| 7 | L12H9 | +0.030 | +0.008 |
| 8 | L17H3 | +0.030 | +0.023 |
| 9 | L19H7 | +0.030 | +0.031 |
| 10 | L13H3 | +0.028 | -0.007 |
| 11 | L12H14 | +0.028 | -0.002 |
| 12 | L17H10 | +0.022 | -0.000 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L18H12 | L18H6; L18H5; L18H15; L18H8; L18H2 |
| 2 | L18H12, L6H0 | L18H10, L6H13; L18H7, L6H11; L18H10, L6H2; L18H6, L6H2; L18H9, L6H4 |
| 4 | L18H12, L6H0, L9H3, L18H14 | L18H11, L6H1, L9H7, L18H3; L18H1, L6H15, L9H0, L18H2; L18H3, L6H13, L9H4, L18H13; L18H7, L6H15, L9H0, L18H4; L18H9, L6H15, L9H9, L18H10 |
