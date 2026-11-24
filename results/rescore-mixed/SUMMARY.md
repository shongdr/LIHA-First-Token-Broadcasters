# Why mean ablation keeps the LCB drop but not the FLORES effect

Post hoc rescoring of saved outputs; nothing is regenerated. LCB: results/<model>-lcb/samples.jsonl.gz (the paper's
runs: the recomputed dLPR values match Tables 4-5 of the submission). FLORES: results/<model>-diag/gens.jsonl.gz
(section 10). For each head, the LCB task where its zero-ablation drop is larger.

- LCB kept: mean-ablation dLPR / zero-ablation dLPR.
- New failures under mean: baseline-passing replies that fail under mean ablation, mixed (some lines in the
  expected language) / whole (none).
- LCB whole-only kept: wholly wrong-language failures under mean / under zero ablation, a FLORES-like count.
- FLORES kept: mean / zero c->w with one label per continuation (as in the paper) and with sentence-level,
  LCB-like scoring (rescore_flores_sentences.py).

| head | LCB task | LCB kept | new failures under mean, mixed / whole | LCB whole-only kept | FLORES kept, one label | FLORES kept, sentence level |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B L22H6 | cross | 90% | 15 / 289 | 93% (289/310) | 90% | 90% |
| Qwen2.5-3B L27H13 | monol | 76% | 172 / 161 | 67% (161/241) | 66% | 76% |
| OLMo-2-1B L12H8 | cross | 43% | 125 / 58 | 26% (58/223) | 70% | 69% |
| Qwen2.5-7B L19H1 | cross | 47% | 24 / 23 | 43% (23/54) | 10% | n/a (see note) |
| Qwen3-1.7B L18H12 | cross | 45% | 31 / 254 | 47% (254/543) | 9% | 18% |
| Gemma-3-1B L11H3 | monol | 39% | 128 / 73 | 30% (73/244) | 22% | 29% |
| Gemma-3-4B L24H0 | monol | 46% | 92 / 8 | 11% (8/74) | 5% | 17% |

Reading:
- Gemma-3-1B and Gemma-3-4B: under mean ablation most new LCB failures are mixed replies, which LCB fails and a
  one-label FLORES score does not count. Counting only wholly wrong replies, LCB keeps about as much of the effect
  as FLORES. Scoring explains most of the gap.
- Qwen3-1.7B and Qwen2.5-7B (crosslingual): mean-ablation failures are often entirely in English, and the
  whole-only count still keeps 43-47% against 9-10% on FLORES. Scoring does not explain the gap. These prompts
  name the reply language once in an English instruction, and the heads attend to that name (section 6 of the
  paper), whereas every token of a FLORES prompt is in the target language.
- Qwen2.5-1.5B, Qwen2.5-3B and OLMo-2-1B already keep much of the FLORES effect under mean ablation, so there is
  little gap to explain. In OLMo-2-1B the whole-only LCB share is lower than on FLORES.

Note: sentence-level FLORES scoring keeps segments of 4+ space-separated words. Qwen2.5-7B's ablated
continuations are mostly Chinese, so most of them are dropped and its sentence-level value is not used.

Files: lcb_split.md (every head and task), flores_sentences.md.
