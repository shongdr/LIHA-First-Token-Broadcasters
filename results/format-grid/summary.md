# Checkpoint x input format, FLORES-selected heads zeroed

Same prompts and token ids in every cell. keep: share of non-English prompts continued in the prompt's language without ablation. c->w: share of all prompts in the prompt's language without ablation and in another language with the head zeroed (Eq. 1). chat: user turn only, no system turn; chat+system: Qwen2.5's default English system prompt, the paper's setting. ctrl: largest c->w among the same-layer controls, if run.

| pair | head | n | format | base keep | base c->w | instruct keep | instruct c->w | ctrl (base / instr) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | L22H6 | 2500 | raw | 0.984 | 0.151 | 0.911 | 0.176 | - / - |
| qwen2.5-1.5b | L22H6 | 2500 | chat | 0.912 | 0.108 | 0.982 | 0.259 | - / - |
| qwen2.5-1.5b | L22H6 | 2500 | chat+system | 0.984 | 0.090 | 0.902 | 0.500 | - / - |
