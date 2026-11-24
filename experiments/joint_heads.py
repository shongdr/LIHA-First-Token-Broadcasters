# Joint ablation of the top-ranked LCB-screen heads against layer-matched random head sets, in the 1-7B models
# (for the ARR response; plan in joint_heads_plan.md). It follows the protocol used for the 8B-14B head sets:
#   - eligible heads: dNLL <= 1 and >= 0.9 of English continuations kept in English on the model's 125-prompt FLORES
#     screen (head_rule.md), read from results/<FLORES screen>/{summary,labels}.json as pick_lcbscreen.py does;
#   - eligible heads ranked by their drop on the model's 100-prompt LCB screen (results/<model>-lcbscreen[-mono]),
#     ties broken by smaller dNLL, then layer, then head index;
#   - the top k heads zeroed or mean-ablated together, for k in --ks;
#   - --n-random random sets per k whose j-th head comes from the layer of the j-th top head, drawn among eligible
#     heads outside the top eight (no head twice in a set, no set twice);
#   - mean ablation uses each head's mean over the 2,500 FLORES prompts, as lcb.py does;
#   - LCB is scored as in lcb.py; the screen's 100 prompts are left out of every comparison.
# Two prompt sets, each with its own baseline run and batching, so every comparison is paired within one set:
#   primary = the screen's task (crosslingual, monolingual for the two Gemma models), non-English, screen prompts out:
#             baseline, top-k sets and random sets;
#   checks  = the other task (non-English) and the English monolingual prompts: baseline and top-k sets only.
# The top-k sets also get the change in FLORES dev loss (100 sentences per language, plain text), as followup.py does.
#
# Steps, from the repo root:
#   python experiments/joint_heads.py plan     --model qwen3-1.7b-instruct   # no GPU: ranking and sets -> plan.md
#   python experiments/joint_heads.py selftest --model qwen3-1.7b-instruct   # GPU: set hook == followup.patched
#   python experiments/joint_heads.py run      --model qwen3-1.7b-instruct [--smoke]
#   python experiments/joint_heads.py report   --model qwen3-1.7b-instruct   # no GPU: summary.md / summary.json
# run resumes: conditions already in samples.jsonl are skipped. A full run uses the plan committed in
# results/<model>-joint/plan.json and stops if the current data would give different sets.
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# FLORES screen, LCB screen, screen task, and the batch size / token budget of the model's LCB runs
CONFIG = {
    "qwen-instruct": ("qwen-instruct", "qwen-instruct-lcbscreen", "crosslingual", 125, 24000, "L22H6"),
    "qwen2.5-3b-instruct": ("qwen2.5-3b-instruct-screen", "qwen2.5-3b-instruct-lcbscreen", "crosslingual",
                            250, 8000, "L27H13"),
    "qwen2.5-7b-instruct": ("qwen2.5-7b-instruct-screen", "qwen2.5-7b-instruct-lcbscreen", "crosslingual",
                            125, 4000, "L19H1"),
    "qwen3-1.7b-instruct": ("qwen3-1.7b-instruct-screen", "qwen3-1.7b-instruct-lcbscreen", "crosslingual",
                            250, 4000, "L18H12"),
    "qwen3-4b-instruct": ("qwen3-4b-instruct-screen", "qwen3-4b-instruct-lcbscreen", "crosslingual",
                          125, 4000, "L24H27"),
    "gemma3-1b-instruct": ("gemma3-1b-instruct-screen", "gemma3-1b-instruct-lcbscreen-mono", "monolingual",
                           250, 6000, "L11H3"),
    "gemma3-4b-instruct": ("gemma3-4b-instruct-screen", "gemma3-4b-instruct-lcbscreen-mono", "monolingual",
                           64, 2000, "L24H0"),
    "olmo2-1b-instruct": ("olmo2-1b-instruct-screen", "olmo2-1b-instruct-lcbscreen", "crosslingual",
                          250, 8000, "L12H8"),
    "llama3.2-1b-instruct": ("llama3.2-1b-instruct-screen", "llama3.2-1b-instruct-lcbscreen", "crosslingual",
                             250, 8000, "L8H25"),
    "llama3.2-3b-instruct": ("llama3.2-3b-instruct-screen", "llama3.2-3b-instruct-lcbscreen", "crosslingual",
                             250, 4000, "L13H19"),
    # not counted in the paper; optional, as the 12-14B models without a qualifying head were
    "olmo3-7b-instruct": ("olmo3-7b-instruct-screen", "olmo3-7b-instruct-lcbscreen", "crosslingual",
                          125, 4000, "L14H25"),
    "smollm3-instruct": ("smollm3-instruct-screen", "smollm3-instruct-lcbscreen", "crosslingual",
                         125, 4000, None),
}
SCREEN_LANGS = ["fr", "de", "es", "it"]
LCB_LANGS = ["fr", "de", "es", "it", "en"]
TOP_POOL = 8  # random heads are drawn outside the top eight


def parse(h):
    l, k = h[1:].split("H")
    return int(l), int(k)


def name(lh):
    return f"L{lh[0]}H{lh[1]}"


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


# ---------------------------------------------------------------------------------------------------------------
# plan: eligibility, ranking and sets (standard library and numpy only)
# ---------------------------------------------------------------------------------------------------------------

def eligibility(flores_dir):
    """dNLL and English retention per head on the 125-prompt FLORES screen, as pick_lcbscreen.py computes them."""
    table = json.load(open(flores_dir / "summary.json", encoding="utf-8"))["modes"]["head"]["table"]
    labels = json.load(open(flores_dir / "labels.json", encoding="utf-8"))
    groups = {}
    for r in csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")):
        groups.setdefault(r["language"], []).append(r["language"])
    expected = [l for g in groups.values() for l in g[:25]]
    assert len(labels["base"]["labels"]) == len(expected) == 125
    en = [i for i, e in enumerate(expected) if e == "en"]
    out = {}
    for h, row in table.items():
        kept = sum(same(labels[f"head:{h}"]["labels"][i], "en") for i in en) / len(en)
        out[h] = {"dnll": row["dnll"], "en_kept": kept, "eligible": row["dnll"] <= 1 and kept >= 0.9}
    return out


def make_plan(model, ks, n_random, seed):
    flores_screen, lcb_screen, task, _, _, paper_head = CONFIG[model]
    elig = eligibility(Path("results") / flores_screen)
    s = json.load(open(Path("results") / lcb_screen / "screen.json", encoding="utf-8"))
    assert set(s["heads"]) == set(elig), "LCB screen and FLORES screen cover different heads"
    drops = {h: s["base"] - r for h, r in s["heads"].items()
             if elig[h]["eligible"] and not (isinstance(r, float) and math.isnan(r))}
    ranked = sorted(drops, key=lambda h: (-round(drops[h], 9), elig[h]["dnll"], *parse(h)))
    top8 = set(ranked[:TOP_POOL])
    by_layer = defaultdict(list)
    for h in drops:
        if h not in top8:
            by_layer[parse(h)[0]].append(h)
    notes = []
    sets = {}
    for k in ks:
        top = ranked[:k]
        if len(ranked) > k and round(drops[ranked[k - 1]], 9) == round(drops[ranked[k]], 9):
            notes.append(f"k={k}: the cut falls inside a tie on the screen ({ranked[k - 1]} and {ranked[k]} both "
                         f"{drops[ranked[k]]:+.3f}); the dNLL tie rule decides membership.")
        rand, seen = [], set()
        for sidx in range(n_random):
            rng = random.Random(seed + 1000 * k + sidx)
            for attempt in range(200):
                used, pick, moved = set(), [], []
                for h in top:
                    l = parse(h)[0]
                    pool = sorted(x for x in by_layer[l] if x not in used)
                    if not pool:  # no eligible head left in that layer: nearest layer with one, recorded
                        near = sorted((abs(L - l), L) for L, v in by_layer.items() if any(x not in used for x in v))
                        L = near[0][1]
                        pool = sorted(x for x in by_layer[L] if x not in used)
                        moved.append(f"{h}->layer {L}")
                    x = rng.choice(pool)
                    used.add(x)
                    pick.append(x)
                key = tuple(sorted(pick))
                if key not in seen:
                    break
            if key in seen:
                notes.append(f"k={k}: only {len(rand)} distinct random sets exist under the layer rule.")
                break
            seen.add(key)
            rand.append(pick)
            if moved:
                notes.append(f"k={k}, random set {sidx}: " + ", ".join(moved))
        sets[k] = {"top": top, "random": rand}
    rank_of_paper = ranked.index(paper_head) + 1 if paper_head in ranked else None
    return {
        "model": model, "task": task, "flores_screen": flores_screen, "lcb_screen": lcb_screen,
        "screen_base": s["base"], "n_heads": len(s["heads"]), "n_eligible": len(drops),
        "ranking": [{"head": h, "drop": drops[h], "dnll": elig[h]["dnll"]} for h in ranked[:12]],
        "paper_head": paper_head, "paper_head_rank": rank_of_paper,
        "ks": ks, "n_random": n_random, "seed": seed, "sets": {str(k): v for k, v in sets.items()}, "notes": notes,
    }


def write_plan(plan, out):
    lines = [f"# {plan['model']}: joint-ablation plan", "",
             f"LCB screen {plan['lcb_screen']} ({plan['task']}, baseline {plan['screen_base']:.3f}); eligible heads "
             f"{plan['n_eligible']} of {plan['n_heads']} (FLORES screen {plan['flores_screen']}). Paper head "
             f"{plan['paper_head']}: rank {plan['paper_head_rank']} among eligible heads.", "",
             "| rank | head | screen drop | dNLL |", "|---|---|---|---|"]
    lines += [f"| {i + 1} | {r['head']} | {r['drop']:+.3f} | {r['dnll']:+.3f} |" for i, r in enumerate(plan["ranking"])]
    lines += ["", "| k | top set | random sets (layer-matched) |", "|---|---|---|"]
    for k, v in plan["sets"].items():
        lines.append(f"| {k} | {', '.join(v['top'])} | " + "; ".join(", ".join(r) for r in v["random"]) + " |")
    if plan["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in plan["notes"]]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(plan, open(out / "plan.json", "w", encoding="utf-8"), indent=1)
    (out / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def conditions(plan):
    """(set, cond name, heads, ablation, k, kind) in run order; the baselines come first."""
    conds = [("primary", "base", [], None, 0, "base"), ("checks", "base", [], None, 0, "base")]
    for k in plan["ks"]:
        v = plan["sets"][str(k)]
        for abl in ("zero", "mean"):
            conds.append(("primary", f"top{k} {abl}", v["top"], abl, k, "top"))
            conds.append(("checks", f"top{k} {abl}", v["top"], abl, k, "top"))
            for i, r in enumerate(v["random"]):
                conds.append(("primary", f"rand{k}.{i} {abl}", r, abl, k, "random"))
    return conds


# ---------------------------------------------------------------------------------------------------------------
# run and selftest (need torch, transformers, fasttext; import the repo's own helpers)
# ---------------------------------------------------------------------------------------------------------------

@contextmanager
def patched_set(model, key, heads, dh, fns):
    """Replace several heads' slices at the o_proj input, one pre-hook per layer; the same edit as
    followup.patched, which handles one head."""
    from sweep import blocks
    by_layer = defaultdict(list)
    for (l, h), fn in zip(heads, fns):
        by_layer[l].append((h, fn))
    handles = []
    for l, hs in by_layer.items():
        def hook(m, args, hs=hs):
            x = args[0].clone()
            for h, fn in hs:
                s = slice(h * dh, (h + 1) * dh)
                x[..., s] = fn(x[..., s])
            return (x,) + args[1:]
        handles.append(blocks(model, key)[l][1].register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def load(model_key, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sweep import MODELS, load_kwargs
    name_, default = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name_)
    tok.pad_token = tok.pad_token or tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name_, dtype=getattr(torch, dtype or default),
                                                 **load_kwargs(model_key)).to(device).eval()
    cfg = model.config.get_text_config()
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return tok, model, dh


def selftest(a):
    """The set hook must reproduce followup.patched exactly: one head, two heads in a layer, two layers, mean."""
    import torch
    from followup import patched, prompt_means
    from sweep import as_prompts
    tok, model, dh = load(a.model, a.dtype)
    rows = list(csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")))[::500][:5]
    tok.padding_side = "left"
    b = tok(as_prompts(tok, a.model, rows), return_tensors="pt", padding=True).to(model.device)
    plan = json.load(open(out_dir(a) / "plan.json", encoding="utf-8"))
    top = [parse(h) for h in plan["sets"][str(max(plan["ks"]))]["top"]]
    H = model.config.get_text_config().num_attention_heads
    l0, h0 = top[0]
    same_layer = [top[0], (l0, (h0 + 1) % H)]  # always test two heads of one layer
    other = next((lh for lh in top if lh[0] != l0), None)
    mu = prompt_means(model, tok, a.model, as_prompts(tok, a.model, rows), top, dh, 8)
    zero = lambda x: torch.zeros_like(x)  # noqa: E731

    @torch.no_grad()
    def logits():
        return model(**b).logits.float()

    base = logits()
    # kernels can differ in the last bits between two identical passes; measure that first and allow twice it
    tol = 2 * (logits() - base).abs().max().item()
    print(f"repeat-pass max |diff| {tol / 2:.3g}; tolerance {tol:.3g}")

    def eq(x, y):
        return (x - y).abs().max().item() <= tol

    checks = []

    def nested(specs):
        from contextlib import ExitStack
        with ExitStack() as st:
            for (l, h), fn in specs:
                st.enter_context(patched(model, a.model, l, h, dh, fn))
            return logits()

    def via_set(specs):
        with patched_set(model, a.model, [lh for lh, _ in specs], dh, [fn for _, fn in specs]):
            return logits()

    with patched_set(model, a.model, [], dh, []):
        checks.append(("empty set leaves the model unchanged", eq(logits(), base)))
    cases = [("one head, zero", [(top[0], zero)]),
             ("two heads in one layer, zero", [(same_layer[0], zero), (same_layer[1], zero)])]
    if other:
        cases.append(("two layers, zero", [(top[0], zero), (other, zero)]))
    m0 = mu[top[0]]
    cases.append(("one head, mean", [(top[0], lambda x, m=m0: m.to(x.dtype).expand_as(x))]))
    for label, specs in cases:
        ref, got = nested(specs), via_set(specs)
        checks.append((f"{label}: equals followup.patched", eq(ref, got)))
        checks.append((f"{label}: changes the logits", (ref - base).abs().max().item() > 10 * tol + 1e-4))
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


def screen_keys(items_all, task):
    """The LCB screen's 100 prompts: the saved list for crosslingual, rebuilt by lcb.py's sampling rule otherwise."""
    if task == "crosslingual":
        saved = json.load(open("results/lcb-screen-heldout/screen-prompts.json", encoding="utf-8"))
        keys = {tuple(x) for x in saved}
    else:
        items = [it for it in items_all if it["task"] == task and it["language"] in SCREEN_LANGS]
        rng, groups = random.Random(0), {}
        for it in items:
            groups.setdefault((it["task"], it["language"]), []).append(it)
        keys = {(it["task"], it["source"], it["language"], it["prompt"])
                for g in groups.values() for it in rng.sample(g, min(25, len(g)))}
    assert len(keys) == 100, len(keys)
    return keys


def rebuild_check(items_all):
    """Rebuilding the crosslingual screen with lcb.py's rule must give the saved list, which validates the rule
    used for the monolingual screens."""
    items = [it for it in items_all if it["task"] == "crosslingual" and it["language"] in SCREEN_LANGS]
    rng, groups = random.Random(0), {}
    for it in items:
        groups.setdefault((it["task"], it["language"]), []).append(it)
    rebuilt = {(it["task"], it["source"], it["language"], it["prompt"])
               for g in groups.values() for it in rng.sample(g, min(25, len(g)))}
    return rebuilt == screen_keys(items_all, "crosslingual")


def out_dir(a):
    return Path(a.out or f"out/{a.model}-joint" + ("-smoke" if getattr(a, "smoke", False) else ""))


def run(a):
    import fasttext
    import torch
    from urllib.request import urlretrieve
    import lcb
    from followup import prompt_means
    from sweep import as_prompts, generate, nll
    from prompts import flores

    out = out_dir(a)
    plan_path = out / "plan.json"
    committed = Path("results") / f"{a.model}-joint" / "plan.json"
    if not a.smoke and committed.exists():
        fixed = json.load(open(committed, encoding="utf-8"))
        again = make_plan(a.model, fixed["ks"], fixed["n_random"], fixed["seed"])
        if again["sets"] != fixed["sets"]:
            sys.exit(f"{committed} no longer matches the screens in results/; not running")
        out.mkdir(parents=True, exist_ok=True)
        json.dump(fixed, open(plan_path, "w", encoding="utf-8"), indent=1)
    elif not plan_path.exists():
        write_plan(make_plan(a.model, a.ks, a.n_random, a.seed), out)
    plan = json.load(open(plan_path, encoding="utf-8"))
    _, _, task, bs, budget, _ = CONFIG[a.model]
    bs, budget = a.bs or bs, a.token_budget or budget

    items = lcb.load_lcb(a.lcb, LCB_LANGS)
    assert rebuild_check(items), "lcb.py's sampling rule no longer reproduces the saved screen prompts"
    skeys = screen_keys(items, task)
    key = lambda it: (it["task"], it["source"], it["language"], it["prompt"])  # noqa: E731
    other = "monolingual" if task == "crosslingual" else "crosslingual"
    sets = {
        "primary": [i for i, it in enumerate(items) if it["task"] == task and it["language"] != "en"
                    and key(it) not in skeys],
        "checks": [i for i, it in enumerate(items) if key(it) not in skeys and
                   ((it["task"] == other and it["language"] != "en") or
                    (it["task"] == "monolingual" and it["language"] == "en"))],
    }
    if a.smoke:
        rng = random.Random(0)
        sets = {s: sorted(rng.sample(v, min(n, len(v)))) for (s, v), n in zip(sets.items(), (24, 12))}
    json.dump({"sets": sets, "items": items}, open(out / "items.json", "w", encoding="utf-8"), ensure_ascii=False)

    if not Path(a.lid).exists():
        urlretrieve(lcb.LID_URL, a.lid)
    lid = fasttext.load_model(a.lid)
    tok, model, dh = load(a.model, a.dtype)
    prompts = as_prompts(tok, a.model, items)
    lengths = [len(tok(p).input_ids) for p in prompts]
    groups = {s: lcb.batches(lengths, sorted(v, key=lambda i: lengths[i]), bs, budget) for s, v in sets.items()}

    heads = sorted({parse(h) for v in plan["sets"].values() for hs in [v["top"], *v["random"]] for h in hs})
    rows = list(csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")))
    mu = prompt_means(model, tok, a.model, as_prompts(tok, a.model, rows), heads, dh, 64)
    zero = lambda x: torch.zeros_like(x)  # noqa: E731
    dev = flores("dev")
    loss_sents = {l: [x.strip() for x in dev[l][:100]] for l in ["en", "fr", "de", "es", "it"]}

    done = set()
    samples = out / "samples.jsonl"
    if samples.exists():
        for line in open(samples, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["set"], r["cond"]))
    conds = [c for c in conditions(plan) if not a.smoke or c[4] in (0, 1, 2)]
    if a.smoke:
        conds = [c for c in conds if c[5] != "random" or c[1].split(".")[1].startswith("0")]
    todo = [c for c in conds if (c[0], c[1]) not in done]
    print(f"{len(conds)} conditions, {len(todo)} to run; primary {len(sets['primary'])} prompts, "
          f"checks {len(sets['checks'])}", flush=True)

    with open(samples, "a", encoding="utf-8") as f, open(out / "nll.jsonl", "a", encoding="utf-8") as fn:
        for set_name, cond, hs, abl, k, kind in todo:
            t0 = time.time()
            lh = [parse(h) for h in hs]
            fns = [zero if abl == "zero" else (lambda x, m=mu[x_]: m.to(x.dtype).expand_as(x)) for x_ in lh]
            texts = {}
            with patched_set(model, a.model, lh, dh, fns):
                for g in groups[set_name]:
                    texts.update(_gen(generate, model, tok, prompts, g, a.max_new_tokens))
                losses = ({l: nll(model, tok, s, "") for l, s in loss_sents.items()}
                          if set_name == "checks" else None)
            for i in sets[set_name]:
                it = items[i]
                sc = lcb.score(texts[i], it["language"], lid)
                f.write(json.dumps({"set": set_name, "cond": cond, "heads": hs, "ablation": abl, "k": k,
                                    "kind": kind, "i": i, "task": it["task"], "source": it["source"],
                                    "language": it["language"], "text": texts[i], **sc,
                                    "rep": lcb.repetition(texts[i])}, ensure_ascii=False) + "\n")
            f.flush()
            if losses is not None:
                fn.write(json.dumps({"cond": cond, "nll": losses}) + "\n")
                fn.flush()
            print(f"{set_name:7s} {cond:16s} {', '.join(hs):40s} {time.time() - t0:7.1f}s", flush=True)


def _gen(generate, model, tok, prompts, g, max_new):
    # lcb.py's run_all: one generate call per token-budget batch, the batch passed as both order and batch size
    texts = generate(model, tok, prompts, g, len(g), max_new)
    return {i: texts[i] for i in g}


# ---------------------------------------------------------------------------------------------------------------
# report (numpy only)
# ---------------------------------------------------------------------------------------------------------------

def boot(d, n=2000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def lpr(recs, idx):
    by = defaultdict(list)
    for i in idx:
        r = recs.get(i)
        if r is not None and not r["skipped"]:
            by[r["source"]].append(r["pass"])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def report(a):
    out = out_dir(a)
    plan = json.load(open(out / "plan.json", encoding="utf-8"))
    data = defaultdict(dict)
    for line in open(out / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        data[(r["set"], r["cond"])][r["i"]] = r
    nll = {}
    if (out / "nll.jsonl").exists():
        for line in open(out / "nll.jsonl", encoding="utf-8"):
            r = json.loads(line)
            nll[r["cond"]] = float(np.mean(list(r["nll"].values())))

    def scored(recs, i):
        return i in recs and not recs[i]["skipped"]

    def delta(set_name, cond, idx):
        base, c = data[(set_name, "base")], data[(set_name, cond)]
        both = [i for i in idx if scored(base, i) and scored(c, i)]
        return boot([c[i]["pass"] - base[i]["pass"] for i in both])

    def diff(set_name, cond_a, conds_b, idx):
        """Paired difference: cond_a minus the mean of conds_b, over prompts scored in all of them and in base."""
        recs = [data[(set_name, c)] for c in [cond_a, *conds_b]] + [data[(set_name, "base")]]
        both = [i for i in idx if all(scored(r, i) for r in recs)]
        A, B = recs[0], recs[1:-1]
        return boot([A[i]["pass"] - np.mean([b[i]["pass"] for b in B]) for i in both])

    task = plan["task"]
    other = "monolingual" if task == "crosslingual" else "crosslingual"
    base_p = data[("primary", "base")]
    prim = sorted(base_p)
    chk = data[("checks", "base")]
    chk_other = sorted(i for i, r in chk.items() if r["task"] == other and r["language"] != "en")
    chk_en = sorted(i for i, r in chk.items() if r["language"] == "en")
    res = {"model": plan["model"], "task": task, "rows": {}}
    lines = [f"# {plan['model']}: joint ablation of top LCB-screen heads", "",
             f"Primary task: {task}, {len(prim)} non-English prompts outside the LCB screen; baseline LPR "
             f"{lpr(base_p, prim):.3f}. dLPR: paired change against the baseline (pooled over prompts scored in "
             "both, 95% bootstrap CI). vs random: the set minus the mean of its layer-matched random sets. "
             "beats: random sets whose paired difference from the set has a CI wholly below zero. vs top-1: the set "
             "minus the top-1 head under the same ablation. Checks: the other task, English monolingual LPR, and "
             "dNLL (mean over five languages of FLORES dev loss change, nats).", "",
             f"| set | heads | ablation | {task[:5]} dLPR | random sets dLPR (range) | vs random | beats | vs top-1 |"
             f" {other[:5]} dLPR | English LPR | dNLL |",
             "|" + "---|" * 11]
    base_nll = nll.get("base")
    base_en = lpr(chk, chk_en)
    for k in plan["ks"]:
        v = plan["sets"][str(k)]
        for abl in ("zero", "mean"):
            cond = f"top{k} {abl}"
            if ("primary", cond) not in data:
                continue
            rconds = [f"rand{k}.{i} {abl}" for i in range(len(v["random"])) if ("primary", f"rand{k}.{i} {abl}") in data]
            d_top = delta("primary", cond, prim)
            d_r = [delta("primary", c, prim)[0] for c in rconds]
            vs_r = diff("primary", cond, rconds, prim) if rconds else (float("nan"),) * 3
            beats = sum(diff("primary", cond, [c], prim)[2] < 0 for c in rconds)
            vs1 = diff("primary", cond, [f"top1 {abl}"], prim) if k != 1 and ("primary", f"top1 {abl}") in data \
                else (float("nan"),) * 3
            row = {"heads": v["top"], "dLPR": d_top, "random_dLPR": d_r, "vs_random": vs_r, "beats": beats,
                   "n_random": len(rconds), "vs_top1": vs1}
            if ("checks", cond) in data:
                row["other_dLPR"] = delta("checks", cond, chk_other)
                row["en_LPR"] = lpr(data[("checks", cond)], chk_en)
                row["dNLL"] = nll[cond] - base_nll if cond in nll and base_nll is not None else float("nan")
            res["rows"][cond] = row
            rr = f"{min(d_r):+.3f} to {max(d_r):+.3f}" if d_r else "-"
            lines.append(
                f"| top{k} | {', '.join(v['top'])} | {abl} | {fmt(d_top)} | {rr} | "
                f"{fmt(vs_r) if rconds else '-'} | {beats}/{len(rconds)} | {fmt(vs1) if k != 1 else '-'} | "
                + (f"{fmt(row['other_dLPR'])} | {row['en_LPR']:.3f} (base {base_en:.3f}) | {row['dNLL']:+.3f} |"
                   if "other_dLPR" in row else "- | - | - |"))
    if plan["notes"]:
        lines += ["", "Plan notes:"] + [f"- {n}" for n in plan["notes"]]
    json.dump(res, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["plan", "selftest", "run", "report"])
    p.add_argument("--model", required=True, choices=list(CONFIG))
    p.add_argument("--ks", default="1,2,4", help="set sizes; the 12-14B runs used 1,2,4,8")
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--smoke", action="store_true", help="36 prompts, k=1,2, one random set: timing and plumbing")
    p.add_argument("--bs", type=int, default=None, help="default: the model's LCB run")
    p.add_argument("--token-budget", type=int, default=None, help="default: the model's LCB run")
    p.add_argument("--dtype", default=None, help="default: sweep.MODELS")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--lcb", default="lcb/test_sets")
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    a.ks = [int(k) for k in a.ks.split(",")]
    if a.step == "plan":
        write_plan(make_plan(a.model, a.ks, a.n_random, a.seed), out_dir(a))
    elif a.step == "selftest":
        if not (out_dir(a) / "plan.json").exists():
            write_plan(make_plan(a.model, a.ks, a.n_random, a.seed), out_dir(a))
        selftest(a)
    elif a.step == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
