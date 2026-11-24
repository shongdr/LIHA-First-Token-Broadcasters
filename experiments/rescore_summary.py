# One table from rescore_mixed_lcb.py and rescore_flores_sentences.py: why mean ablation keeps much of the LCB drop
# but little of the FLORES effect in four models. Run both scripts first; from the repo root:
#   python experiments/rescore_summary.py [output dir]
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/rescore-mixed")
MEAN = "mean of all prompt tokens (follow-up)"
ROWS = [("qwen-instruct", "L22H6", "crosslingual", "Qwen2.5-1.5B"),
        ("qwen2.5-3b-instruct", "L27H13", "monolingual", "Qwen2.5-3B"),
        ("olmo2-1b-instruct", "L12H8", "crosslingual", "OLMo-2-1B"),
        ("qwen2.5-7b-instruct", "L19H1", "crosslingual", "Qwen2.5-7B"),
        ("qwen3-1.7b-instruct", "L18H12", "crosslingual", "Qwen3-1.7B"),
        ("gemma3-1b-instruct", "L11H3", "monolingual", "Gemma-3-1B"),
        ("gemma3-4b-instruct", "L24H0", "monolingual", "Gemma-3-4B")]

lcb = json.load(open(OUT / "lcb_split.json", encoding="utf-8"))
flo = json.load(open(OUT / "flores_sentences.json", encoding="utf-8"))
lines = [
    "# Why mean ablation keeps the LCB drop but not the FLORES effect", "",
    "Post hoc rescoring of saved outputs; nothing is regenerated. LCB: results/<model>-lcb/samples.jsonl.gz (the paper's",
    "runs: the recomputed dLPR values match Tables 4-5 of the submission). FLORES: results/<model>-diag/gens.jsonl.gz",
    "(section 10). For each head, the LCB task where its zero-ablation drop is larger.", "",
    "- LCB kept: mean-ablation dLPR / zero-ablation dLPR.",
    "- New failures under mean: baseline-passing replies that fail under mean ablation, mixed (some lines in the",
    "  expected language) / whole (none).",
    "- LCB whole-only kept: wholly wrong-language failures under mean / under zero ablation, a FLORES-like count.",
    "- FLORES kept: mean / zero c->w with one label per continuation (as in the paper) and with sentence-level,",
    "  LCB-like scoring (rescore_flores_sentences.py).", "",
    "| head | LCB task | LCB kept | new failures under mean, mixed / whole | LCB whole-only kept |"
    " FLORES kept, one label | FLORES kept, sentence level |",
    "|---|---|---|---|---|---|---|"]
for key, head, task, name in ROWS:
    z, m = lcb[f"{key}-lcb|{head}|{task}|zero"], lcb[f"{key}-lcb|{head}|{task}|mean"]
    fz, fm = flo[key]["zero"], flo[key][MEAN]
    strict = ("n/a (see note)" if key == "qwen2.5-7b-instruct"
              else f"{fm['c2w_strict'] / fz['c2w_strict']:.0%}")
    lines.append(f"| {name} {head} | {task[:5]} | {m['dLPR_pooled'] / z['dLPR_pooled']:.0%} | "
                 f"{m['new_mixed']} / {m['new_whole']} | {m['new_whole'] / z['new_whole']:.0%} "
                 f"({m['new_whole']}/{z['new_whole']}) | {fm['c2w_single'] / fz['c2w_single']:.0%} | {strict} |")
lines += [
    "", "Reading:",
    "- Gemma-3-1B and Gemma-3-4B: under mean ablation most new LCB failures are mixed replies, which LCB fails and a",
    "  one-label FLORES score does not count. Counting only wholly wrong replies, LCB keeps about as much of the effect",
    "  as FLORES. Scoring explains most of the gap.",
    "- Qwen3-1.7B and Qwen2.5-7B (crosslingual): mean-ablation failures are often entirely in English, and the",
    "  whole-only count still keeps 43-47% against 9-10% on FLORES. Scoring does not explain the gap. These prompts",
    "  name the reply language once in an English instruction, and the heads attend to that name (section 6 of the",
    "  paper), whereas every token of a FLORES prompt is in the target language.",
    "- Qwen2.5-1.5B, Qwen2.5-3B and OLMo-2-1B already keep much of the FLORES effect under mean ablation, so there is",
    "  little gap to explain. In OLMo-2-1B the whole-only LCB share is lower than on FLORES.",
    "", "Note: sentence-level FLORES scoring keeps segments of 4+ space-separated words. Qwen2.5-7B's ablated",
    "continuations are mostly Chinese, so most of them are dropped and its sentence-level value is not used.",
    "", "Files: lcb_split.md (every head and task), flores_sentences.md."]
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
