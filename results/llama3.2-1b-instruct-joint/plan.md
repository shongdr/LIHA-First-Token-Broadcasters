# llama3.2-1b-instruct: joint-ablation plan

LCB screen llama3.2-1b-instruct-lcbscreen (crosslingual, baseline 0.900); eligible heads 512 of 512 (FLORES screen llama3.2-1b-instruct-screen). Paper head L8H25: rank 1 among eligible heads.

| rank | head | screen drop | dNLL |
|---|---|---|---|
| 1 | L8H25 | +0.710 | +0.067 |
| 2 | L13H4 | +0.100 | +0.038 |
| 3 | L6H31 | +0.070 | +0.008 |
| 4 | L9H13 | +0.070 | +0.024 |
| 5 | L3H1 | +0.041 | -0.000 |
| 6 | L6H15 | +0.040 | -0.001 |
| 7 | L6H11 | +0.031 | -0.001 |
| 8 | L4H24 | +0.030 | +0.012 |
| 9 | L9H9 | +0.030 | +0.018 |
| 10 | L6H24 | +0.021 | +0.001 |
| 11 | L0H5 | +0.021 | +0.007 |
| 12 | L5H19 | +0.021 | +0.008 |

| k | top set | random sets (layer-matched) |
|---|---|---|
| 1 | L8H25 | L8H9; L8H26; L8H19; L8H28; L8H2 |
| 2 | L8H25, L13H4 | L8H13, L13H16; L8H3, L13H13; L8H13, L13H22; L8H29, L13H22; L8H6, L13H26 |
| 4 | L8H25, L13H4, L6H31, L9H13 | L8H15, L13H0, L6H6, L9H21; L8H11, L13H20, L6H1, L9H2; L8H21, L13H17, L6H27, L9H17; L8H3, L13H20, L6H0, L9H24; L8H5, L13H20, L6H13, L9H19 |
