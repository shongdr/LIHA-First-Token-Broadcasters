# Top-ranked heads ablated together in the 1-7B models

Written 2026-11-25, 08:09 KST, before the full runs. A selftest and a smoke run on Qwen2.5-1.5B (below) were done
before this note; they check the code, and their numbers are not results.

## Why

The paper ablates one head at a time. A reviewer asks whether the selected head is only a bottleneck in a larger
multi-head circuit, which single-head ablation cannot show. The revision answers this for six 8B-14B models with
top-k head sets against layer-matched random sets. This applies the same protocol to the 1-7B models of the main
analysis, so that both size groups have head-set results.

## What and how

- Models: the ten instruct models with a counted head (Qwen2.5-1.5B, 3B, 7B, Qwen3-1.7B, 4B, Gemma-3-1B, 4B,
  OLMo-2-1B, Llama-3.2-1B, 3B). OLMo-3-7B and SmolLM3-3B are optional and reported separately if run.
- Eligible heads: dNLL at most 1 and at least 0.9 of English continuations kept in English on the model's 125-prompt
  FLORES screen (head_rule.md), computed as pick_lcbscreen.py does.
- Ranking: each eligible head's drop on the model's 100-prompt LCB screen (results/<model>-lcbscreen; the
  monolingual screen results/<model>-lcbscreen-mono for the two Gemma models, as in section 9e). Ties are broken by
  smaller dNLL, then layer, then head index.
- Sets: the top k heads for k = 1, 2, 4. For each k, five random sets whose j-th head comes from the layer of the
  j-th top head, drawn among eligible heads outside the top eight, with no head twice in a set and no set twice
  (seed 20261125). Where a layer has too few such heads (Gemma-3-1B has four per layer), fewer distinct sets exist and
  the plan says so. The rankings and sets are fixed in results/<model>-joint/plan.json and plan.md, committed with
  this note; `run` stops if the current screens would give different sets.
- Ablations: zero, and mean (each head's mean over the prompt tokens of the 2,500 FLORES prompts, as in lcb.py),
  applied to all heads of a set at once at the input of the attention output projection.
- Prompts, each set with its own baseline run and batching so that every comparison is paired:
  - primary: the screen's task (crosslingual; monolingual for Gemma), non-English, with the screen's 100 prompts left
    out (1,096 crosslingual or 700 monolingual): baseline, top-k sets and random sets;
  - checks: the other task (non-English) and the 200 English monolingual prompts: baseline and top-k sets, plus the
    change in loss on 100 FLORES dev sentences per language (plain text, as followup.py).
- Generation and scoring as in lcb.py: greedy, 100 new tokens, chat template, batch size, token budget and precision
  of the model's existing LCB run, line-level pass rate with fastText.
- Code: joint_heads.py (plan, selftest, run, report); joint_heads_queue.sh runs the ten models, one queue per GPU.

## Readings, fixed now

For each model, k and ablation, on the primary prompts:
1. The set's paired change in LPR against the baseline, with a 2,000-resample bootstrap 95% CI.
2. The set against its random sets: the paired difference from the mean of the random sets, with its CI, and the
   number of random sets whose paired difference from the set has a CI wholly below zero. A set counts as lowering
   adherence beyond its random sets when that difference CI lies below zero and it beats every random set.
3. For k = 2 and 4, the set against the top-1 head under the same ablation, with its CI. A set counts as adding to
   the top-1 head when that CI lies below zero.
4. Checks for the top-k sets: the other task's paired change, English LPR, and dNLL.

These are read as in the 8B-14B analysis. The comparison between the two size groups does not separate size from
model family, and the top-1 head here is the LCB screen's, so in Qwen2.5-1.5B and Qwen2.5-7B it is L16H9 and L0H1,
not the FLORES-selected L22H6 (rank 2) and L19H1 (tied rank 4).

## Checks done before this note

- selftest on Qwen2.5-1.5B-Instruct (fp16, one RTX 5080 Laptop GPU, torch 2.11 cu128, transformers 5.6.2): the set
  hook gives the same logits as followup.patched for one head, two heads in one layer, two layers, and a mean
  replacement (9/9 checks, difference 0).
- Smoke run on the same model (36 prompts, k = 1 and 2, one random set): all 14 conditions ran and report produced
  its table. dNLL matches the FLORES screen's table (L16H9 -0.007; L16H9 with L22H6 +0.227, against +0.228 for L22H6
  alone). About 20 s per condition on 24 prompts on that GPU.
