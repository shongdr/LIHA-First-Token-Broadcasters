# Checkpoint x input format for the seven FLORES-selected heads

Written 2026-11-25 before the runs of the other six pairs. The Qwen2.5-1.5B pair was run first to check the code
against the existing results (below).

## Why

The base-checkpoint comparison changes checkpoint and input format together (base models got plain text, instruct
models their chat template), and in Qwen2.5 the template also adds a default English system prompt, which alone
halves L22H6's effect (results/qwen-system-2500). Only Qwen2.5-1.5B was crossed, and without controlling the system
prompt. This crosses checkpoint and format for all seven FLORES-selected heads, with the system prompt controlled
and the same prompts and token ids in every cell.

## What and how

- Pairs: Qwen2.5-1.5B L22H6, Qwen2.5-3B L27H13, Qwen2.5-7B L19H1, Qwen3-1.7B L18H12, Gemma-3-1B L11H3, Gemma-3-4B
  L24H0, OLMo-2-1B L12H8; base and instruct checkpoints of sweep.MODELS.
- Formats: raw (prompt text); chat (the instruct template, one user turn, no system turn; written by hand for
  Qwen2.5, whose template would add its default system prompt); chat+system for the three Qwen2.5 pairs (the default
  English system prompt, the paper's setting).
- Both checkpoints use the instruct tokenizer's rendering and ids, after checking that the base tokenizer gives the
  same ids on raw text, and the instruct model's precision (App. A).
- Prompts: the 2,500 FLORES prompts of prompts/prompts_european.csv, the set behind the existing base and instruct
  numbers. `--prompts` takes another file and `--skip-screen` drops the 125 screen prompts.
- Head zeroed at the o_proj input (sweep.ablated, mode head); greedy, 40 new tokens, end of text blocked only for the
  OLMo-2-1B base model as in the sweep; langdetect with seed 0; c->w as Eq. 1.
- Code: format_grid.py (run per pair, report across pairs).

## Readings, fixed now

For each pair and format: the unablated keep rate and c->w in each checkpoint. The checkpoint difference is read
within a format. The main reading is chat without a system prompt: whether the instruct checkpoint's c->w exceeds
the base checkpoint's there, and whether that gap is larger than under raw text. Differences of a few prompts are
not read, since fp16/bf16 batches can shift single continuations.

## Check on Qwen2.5-1.5B (run before this note)

2,500 prompts, fp16, one RTX 5080 Laptop GPU, about 25-32 s per condition. Four existing values are reproduced
within 0.001: base raw 0.151 (sweep 0.150), instruct raw 0.176 (qwen-format-2500 0.176), base chat+system 0.090
(0.089), instruct chat+system 0.500 (qwen-system-2500 0.500), instruct chat without system 0.259 (0.258). New cell:
base chat without system 0.108.
