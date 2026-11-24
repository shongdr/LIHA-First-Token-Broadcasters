# FLORES continuations rescored at sentence level

Single: the paper's C->W (one langdetect label per continuation, all 2,500 selection prompts).
Strict: baseline continuations whose every segment of >= 4 words is in the prompt language, turning into continuations with at least one segment in another language, over prompts scored in both runs. Kept: mean ablation as a share of zero ablation. Mixed/whole: new strict failures with / without any segment in the prompt language.

| head | single zero | single mean | kept | strict zero | strict mean | kept | zero mixed/whole | mean mixed/whole |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B L22H6 | 0.501 | 0.451 | 90% | 0.505 | 0.453 | 90% | 76/1185 | 72/1059 |
| Qwen2.5-3B L27H13 | 0.515 | 0.338 | 66% | 0.489 | 0.374 | 76% | 186/889 | 232/688 |
| Qwen2.5-7B L19H1 | 0.202 | 0.021 | 10% | 0.041 | 0.028 | 68% | 39/46 | 53/16 |
| Qwen3-1.7B L18H12 | 0.324 | 0.028 | 9% | 0.355 | 0.064 | 18% | 171/716 | 112/48 |
| Gemma-3-1B L11H3 | 0.413 | 0.089 | 22% | 0.433 | 0.124 | 29% | 211/864 | 169/141 |
| Gemma-3-4B L24H0 | 0.214 | 0.012 | 5% | 0.255 | 0.043 | 17% | 186/452 | 90/18 |
| OLMo-2-1B L12H8 | 0.080 | 0.056 | 70% | 0.101 | 0.069 | 69% | 57/194 | 46/127 |
