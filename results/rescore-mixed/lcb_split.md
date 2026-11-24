# Mixed vs wholly wrong-language LCB failures under zero and mean ablation

New failures: baseline-passing non-English replies that fail under the condition (both scored).
Mixed: some lines in the expected language. Whole: none. Unscored: no 5+ word line after ablation.
dLPR: pooled paired change over prompts scored in both runs (prompt-weighted, as in the paper).

| model | head | task | cond | dLPR | new fails | mixed | whole | mixed share | became unscored | whole replies go to |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct | L22H6 | monol | zero | -0.273 | 224 | 78 | 146 | 0.35 | 1 | es 89, several 31, en 12 |
| qwen-instruct | L22H6 | monol | mean | -0.246 | 203 | 51 | 152 | 0.25 | 1 | es 133, en 11, several 5 |
| qwen-instruct | L22H6 | cross | zero | -0.276 | 338 | 28 | 310 | 0.08 | 4 | en 147, es 111, fr 28 |
| qwen-instruct | L22H6 | cross | mean | -0.247 | 304 | 15 | 289 | 0.05 | 4 | es 137, en 124, fr 26 |
| qwen2.5-3b-instruct | L27H13 | monol | zero | -0.537 | 422 | 181 | 241 | 0.43 | 19 | es 113, several 41, en 39 |
| qwen2.5-3b-instruct | L27H13 | monol | mean | -0.409 | 333 | 172 | 161 | 0.52 | 0 | es 89, several 33, fr 22 |
| qwen2.5-3b-instruct | L27H13 | cross | zero | -0.449 | 531 | 153 | 378 | 0.29 | 6 | en 253, es 62, several 34 |
| qwen2.5-3b-instruct | L27H13 | cross | mean | -0.331 | 395 | 151 | 244 | 0.38 | 4 | en 125, es 77, several 20 |
| qwen3-1.7b-instruct | L18H12 | monol | zero | -0.149 | 124 | 70 | 54 | 0.56 | 0 | en 31, es 13, pt 6 |
| qwen3-1.7b-instruct | L18H12 | monol | mean | -0.067 | 58 | 41 | 17 | 0.71 | 2 | fr 8, es 5, it 4 |
| qwen3-1.7b-instruct | L18H12 | cross | zero | -0.508 | 602 | 59 | 543 | 0.10 | 5 | en 536, several 5, pt 1 |
| qwen3-1.7b-instruct | L18H12 | cross | mean | -0.230 | 285 | 31 | 254 | 0.11 | 4 | en 239, fr 5, it 3 |
| gemma3-1b-instruct | L11H3 | monol | zero | -0.628 | 507 | 263 | 244 | 0.52 | 0 | en 154, ru 40, several 32 |
| gemma3-1b-instruct | L11H3 | monol | mean | -0.244 | 201 | 128 | 73 | 0.64 | 0 | it 21, several 19, es 15 |
| gemma3-1b-instruct | L11H3 | cross | zero | -0.091 | 117 | 27 | 90 | 0.23 | 8 | en 85, several 5 |
| gemma3-1b-instruct | L11H3 | cross | mean | -0.092 | 112 | 33 | 79 | 0.29 | 1 | en 78, es 1 |
| olmo2-1b-instruct | L12H8 | monol | zero | -0.272 | 218 | 92 | 126 | 0.42 | 4 | en 101, fr 13, es 7 |
| olmo2-1b-instruct | L12H8 | monol | mean | -0.128 | 108 | 65 | 43 | 0.60 | 2 | en 15, es 12, fr 10 |
| olmo2-1b-instruct | L12H8 | cross | zero | -0.335 | 400 | 177 | 223 | 0.44 | 5 | en 188, several 16, fr 15 |
| olmo2-1b-instruct | L12H8 | cross | mean | -0.143 | 183 | 125 | 58 | 0.68 | 2 | fr 21, en 19, es 10 |
| llama3.2-1b-instruct | L8H25 | monol | zero | -0.016 | 15 | 10 | 5 | 0.67 | 0 | es 3, pt 2 |
| llama3.2-1b-instruct | L8H25 | monol | mean | -0.015 | 14 | 10 | 4 | 0.71 | 0 | several 1, es 1, pt 1 |
| llama3.2-1b-instruct | L8H25 | cross | zero | -0.711 | 846 | 34 | 812 | 0.04 | 1 | en 788, several 24 |
| llama3.2-1b-instruct | L8H25 | cross | mean | -0.357 | 429 | 57 | 372 | 0.13 | 2 | en 355, several 15, fr 2 |
| olmo3-7b-instruct | L14H25 | monol | zero | -0.041 | 46 | 30 | 16 | 0.65 | 0 | en 15, several 1 |
| olmo3-7b-instruct | L14H25 | monol | mean | +0.001 | 14 | 14 | 0 | 1.00 | 0 |  |
| olmo3-7b-instruct | L14H25 | cross | zero | -0.018 | 53 | 35 | 18 | 0.66 | 3 | en 17, several 1 |
| olmo3-7b-instruct | L14H25 | cross | mean | +0.015 | 20 | 19 | 1 | 0.95 | 1 | en 1 |
| gemma3-4b-instruct | L24H0 | monol | zero | -0.259 | 212 | 138 | 74 | 0.65 | 2 | es 33, several 17, pt 16 |
| gemma3-4b-instruct | L24H0 | monol | mean | -0.120 | 100 | 92 | 8 | 0.92 | 2 | es 8 |
| gemma3-4b-instruct | L24H0 | cross | zero | -0.107 | 127 | 63 | 64 | 0.50 | 0 | en 47, several 12, es 5 |
| gemma3-4b-instruct | L24H0 | cross | mean | -0.053 | 65 | 47 | 18 | 0.72 | 0 | en 16, several 1, es 1 |
| qwen2.5-7b-instruct | L19H1 | monol | zero | -0.009 | 14 | 10 | 4 | 0.71 | 0 | en 3, zh 1 |
| qwen2.5-7b-instruct | L19H1 | monol | mean | +0.003 | 4 | 4 | 0 | 1.00 | 0 |  |
| qwen2.5-7b-instruct | L19H1 | cross | zero | -0.074 | 91 | 37 | 54 | 0.41 | 1 | en 52, sr 1, zh 1 |
| qwen2.5-7b-instruct | L19H1 | cross | mean | -0.035 | 47 | 24 | 23 | 0.51 | 0 | en 23 |
| llama3.2-3b-instruct-lcb-top | L13H19 | monol | zero | +0.000 | 4 | 4 | 0 | 1.00 | 0 |  |
| llama3.2-3b-instruct-lcb-top | L13H19 | monol | mean | -0.005 | 6 | 5 | 1 | 0.83 | 0 | it 1 |
| llama3.2-3b-instruct-lcb-top | L13H19 | cross | zero | -0.335 | 403 | 92 | 311 | 0.23 | 6 | en 292, several 18, fr 1 |
| llama3.2-3b-instruct-lcb-top | L13H19 | cross | mean | -0.250 | 306 | 84 | 222 | 0.27 | 7 | en 211, several 11 |
