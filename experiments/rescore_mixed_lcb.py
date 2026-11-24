"""Split LCB failures under zero vs mean ablation into mixed and wholly-wrong replies.

Reads the saved LCB samples (results/<model>-lcb/samples.jsonl.gz) and, for each selected head,
compares the baseline with zero and mean ablation on the non-English prompts.

For every prompt scored in both the baseline and the condition (as in the paper's paired ΔLPR):
  pass   = every scored line in the expected language
  mixed  = at least one line in the expected language and at least one not
  whole  = no line in the expected language
New failures are baseline passes that fail under the condition, split into mixed and whole.
Replies that become unscored (no line of 5+ words) are counted separately; ΔLPR excludes them.
"""
import ast
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# usage, from the repo root: python experiments/rescore_mixed_lcb.py [results dir] [output dir]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/rescore-mixed")

RUNS = [  # (results folder, selected head, how it was selected)
    ("qwen-instruct-lcb", "L22H6", "FLORES"),
    ("qwen2.5-3b-instruct-lcb", "L27H13", "FLORES"),
    ("qwen3-1.7b-instruct-lcb", "L18H12", "FLORES"),
    ("gemma3-1b-instruct-lcb", "L11H3", "FLORES"),
    ("olmo2-1b-instruct-lcb", "L12H8", "FLORES"),
    ("llama3.2-1b-instruct-lcb", "L8H25", "LCB"),
    ("olmo3-7b-instruct-lcb", "L14H25", "not counted"),
    ("gemma3-4b-instruct-lcb", "L24H0", "FLORES"),
    ("qwen2.5-7b-instruct-lcb", "L19H1", "FLORES"),
    ("llama3.2-3b-instruct-lcb-top", "L13H19", "LCB"),
]
TASKS = ["monolingual", "crosslingual"]


def as_bool(v):
    return v is True or v == "True"


def load(folder):
    rows = defaultdict(list)
    with gzip.open(RESULTS / folder / "samples.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["skipped"] = as_bool(r["skipped"])
            if not r["skipped"]:
                r["labels"] = ast.literal_eval(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
                r["pass"] = as_bool(r["pass"])
            rows[r["cond"]].append(r)
    return rows


def kind(r):
    if r["skipped"]:
        return "unscored"
    hits = sum(l == r["language"] for l in r["labels"])
    if hits == len(r["labels"]):
        return "pass"
    return "mixed" if hits else "whole"


def other_language(r):
    """For a wholly-wrong reply: the single other language if all lines share one, else 'several'."""
    labs = set(r["labels"])
    return labs.pop() if len(labs) == 1 else "several"


def analyse(base, cond, task):
    idx = [i for i, b in enumerate(base) if b["task"] == task and b["language"] != "en"]
    both = [i for i in idx if not base[i]["skipped"] and not cond[i]["skipped"]]
    d = np.mean([cond[i]["pass"] - base[i]["pass"] for i in both])
    bpass = [i for i in idx if kind(base[i]) == "pass"]
    new = defaultdict(int)
    whole_to = defaultdict(int)
    for i in bpass:
        k = kind(cond[i])
        if k != "pass":
            new[k] += 1
            if k == "whole":
                whole_to[other_language(cond[i])] += 1
    fails = new["mixed"] + new["whole"]
    return {
        "n_nonen": len(idx), "n_both": len(both), "dLPR_pooled": float(d),
        "base_pass": len(bpass), "new_fail": fails, "new_mixed": new["mixed"], "new_whole": new["whole"],
        "new_unscored": new["unscored"],
        "mixed_share": new["mixed"] / fails if fails else float("nan"),
        "whole_to": dict(sorted(whole_to.items(), key=lambda kv: -kv[1])),
    }


def main():
    out = {}
    lines = ["# Mixed vs wholly wrong-language LCB failures under zero and mean ablation", "",
             "New failures: baseline-passing non-English replies that fail under the condition (both scored).",
             "Mixed: some lines in the expected language. Whole: none. Unscored: no 5+ word line after ablation.",
             "dLPR: pooled paired change over prompts scored in both runs (prompt-weighted, as in the paper).", ""]
    hdr = ("| model | head | task | cond | dLPR | new fails | mixed | whole | mixed share | became unscored |"
           " whole replies go to |")
    lines += [hdr, "|" + "---|" * 11]
    for folder, head, how in RUNS:
        p = RESULTS / folder / "samples.jsonl.gz"
        if not p.exists():
            continue
        rows = load(folder)
        base = rows["base"]
        for task in TASKS:
            for c in ("zero", "mean"):
                cond = rows.get(f"{head} {c}")
                if cond is None:
                    continue
                r = analyse(base, cond, task)
                out[f"{folder}|{head}|{task}|{c}"] = {"selected_on": how, **r}
                top = ", ".join(f"{k} {v}" for k, v in list(r["whole_to"].items())[:3])
                lines.append(f"| {folder.removesuffix('-lcb')} | {head} | {task[:5]} | {c} | {r['dLPR_pooled']:+.3f} | "
                             f"{r['new_fail']} | {r['new_mixed']} | {r['new_whole']} | {r['mixed_share']:.2f} | "
                             f"{r['new_unscored']} | {top} |")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "lcb_split.json", "w"), indent=1)
    (OUT / "lcb_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
