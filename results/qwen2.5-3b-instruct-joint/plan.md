# qwen2.5-3b-instruct: joint-ablation plan

LCB screen qwen2.5-3b-instruct-lcbscreen (crosslingual, baseline 0.929); eligible heads 576 of 576 (FLORES screen qwen2.5-3b-instruct-screen). Paper head L27H13: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L27H13 | +0.389 | +0.234 |
| 2 | L27H11 | +0.052 | +0.000 |
| 3 | L21H8 | +0.051 | -0.003 |
| 4 | L18H6 | +0.051 | +0.003 |
| 5 | L26H5 | +0.042 | -0.001 |
| 6 | L22H0 | +0.040 | -0.003 |
| 7 | L17H3 | +0.039 | -0.000 |
| 8 | L24H14 | +0.039 | +0.001 |
| 9 | L13H9 | +0.039 | +0.002 |
| 10 | L12H13 | +0.039 | +0.010 |
| 11 | L11H2 | +0.031 | -0.003 |
| 12 | L9H0 | +0.031 | -0.002 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L27H13 | L27H6; L27H5; L27H15; L27H8; L27H2 |
| 2 | L27H13, L27H11 | L27H10, L27H15; L27H7, L27H10; L27H10, L27H4; L27H6, L27H3; L27H9, L27H5 |
| 4 | L27H13, L27H11, L21H8, L18H6 | L27H12, L27H0, L21H6, L18H14; L27H1, L27H3, L21H0, L18H13; L27H3, L27H14, L21H3, L18H12; L27H7, L27H2, L21H0, L18H2; L27H9, L27H2, L21H9, L18H10 |
