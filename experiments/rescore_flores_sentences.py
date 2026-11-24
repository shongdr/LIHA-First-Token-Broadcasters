"""Rescore zero- and mean-ablated FLORES continuations at sentence level.

The paper's C->W gives each 40-token continuation one langdetect label. LCB instead fails a reply if any scored
line is in another language. This script asks whether mean ablation keeps more of the zero-ablation effect on
FLORES when FLORES is scored the strict, LCB-like way.

Input: results/<model>-diag/gens.jsonl.gz (conditions 'base', 'zero', 'mean of all prompt tokens (follow-up)')
and results/<model>-diag/labels.json (the stored whole-continuation labels), from the team repo.
Prompt languages come from prompts/prompts_european.csv (2,500 rows, the order of the texts).

Strict scoring: split a continuation into sentences/lines, keep segments with >= 4 words, label each with langdetect
(seed 0). strict pass = every kept segment in the prompt language; no kept segment = unscored (left out).
  mixed = some kept segments in the prompt language, some not; whole = none.
"""
import csv
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

# usage, from the repo root: python experiments/rescore_flores_sentences.py [repo root] [output dir]
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/rescore-mixed")
MODELS = [("qwen-instruct", "Qwen2.5-1.5B L22H6"), ("qwen2.5-3b-instruct", "Qwen2.5-3B L27H13"),
          ("qwen2.5-7b-instruct", "Qwen2.5-7B L19H1"), ("qwen3-1.7b-instruct", "Qwen3-1.7B L18H12"),
          ("gemma3-1b-instruct", "Gemma-3-1B L11H3"), ("gemma3-4b-instruct", "Gemma-3-4B L24H0"),
          ("olmo2-1b-instruct", "OLMo-2-1B L12H8")]
MEAN = "mean of all prompt tokens (follow-up)"
SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
MIN_WORDS = 4


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


def segments(text):
    return [s.strip() for s in SPLIT.split(text) if len(s.split()) >= MIN_WORDS]


def label_segments(text):
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    out = []
    for s in segments(text):
        try:
            out.append(detect(s))
        except LangDetectException:
            out.append("unknown")
    return out


def strict_kind(labels, expected):
    if not labels:
        return "unscored"
    hits = sum(same(l, expected) for l in labels)
    return "pass" if hits == len(labels) else ("mixed" if hits else "whole")


def main():
    expected = [r["language"] for r in csv.DictReader(open(REPO / "prompts/prompts_european.csv", encoding="utf-8"))]
    assert len(expected) == 2500
    report, rows = {}, []
    with Pool(8) as pool:
        for key, name in MODELS:
            d = REPO / "results" / f"{key}-diag"
            texts = {}
            with gzip.open(d / "gens.jsonl.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["cond"] in ("base", "zero", MEAN):
                        texts[r["cond"]] = r["texts"]
            whole_labels = json.load(open(d / "labels.json", encoding="utf-8"))
            seg = {c: pool.map(label_segments, t, chunksize=50) for c, t in texts.items()}
            res = {}
            for c in ("zero", MEAN):
                # paper metric: one label per continuation
                wl_b, wl_c = whole_labels["base"]["labels"], whole_labels[c]["labels"]
                c2w = sum(same(b, e) and not same(x, e) for b, x, e in zip(wl_b, wl_c, expected)) / 2500
                # strict metric: baseline strict pass -> strict fail, among prompts scored in both
                kb = [strict_kind(s, e) for s, e in zip(seg["base"], expected)]
                kc = [strict_kind(s, e) for s, e in zip(seg[c], expected)]
                both = [i for i in range(2500) if kb[i] != "unscored" and kc[i] != "unscored"]
                new = [kc[i] for i in both if kb[i] == "pass" and kc[i] != "pass"]
                res[c] = {"c2w_single": c2w, "c2w_strict": len(new) / len(both), "n_both": len(both),
                          "new_mixed": new.count("mixed"), "new_whole": new.count("whole")}
            z, m = res["zero"], res[MEAN]
            report[key] = res
            rows.append(f"| {name} | {z['c2w_single']:.3f} | {m['c2w_single']:.3f} | "
                        f"{m['c2w_single'] / z['c2w_single']:.0%} | {z['c2w_strict']:.3f} | {m['c2w_strict']:.3f} | "
                        f"{m['c2w_strict'] / z['c2w_strict']:.0%} | {z['new_mixed']}/{z['new_whole']} | "
                        f"{m['new_mixed']}/{m['new_whole']} |")
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "flores_sentences.json", "w"), indent=1)
    lines = ["# FLORES continuations rescored at sentence level", "",
             "Single: the paper's C->W (one langdetect label per continuation, all 2,500 selection prompts).",
             f"Strict: baseline continuations whose every segment of >= {MIN_WORDS} words is in the prompt language, "
             "turning into continuations with at least one segment in another language, over prompts scored in "
             "both runs. Kept: mean ablation as a share of zero ablation. Mixed/whole: new strict failures with / "
             "without any segment in the prompt language.", "",
             "| head | single zero | single mean | kept | strict zero | strict mean | kept | zero mixed/whole |"
             " mean mixed/whole |", "|" + "---|" * 9] + rows
    (OUT / "flores_sentences.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
