# Checkpoint x input-format grid for the seven FLORES-selected heads (for the ARR response).
#
# Each head is zeroed in its base and instruct checkpoints, on the same FLORES prompts, in each input format:
#   raw          the prompt text alone
#   chat         the instruct model's chat template with one user turn and no system turn
#   chat+system  Qwen2.5 only: the template with its default English system prompt, the paper's setting
# Both checkpoints of a pair get the same token ids: every prompt is rendered with the instruct tokenizer's template
# and tokenized with the instruct tokenizer, after checking that the base tokenizer gives the same ids on raw text.
# Precision is the instruct model's for both checkpoints (App. A). Generation and labels follow the sweep: greedy,
# 40 new tokens, end-of-text blocked only where sweep.py blocks it (OLMo-2-1B base), langdetect with seed 0.
#
# From the repo root:
#   python experiments/format_grid.py run --pair qwen2.5-1.5b [--per-lang 10]   # GPU
#   python experiments/format_grid.py report                                   # no GPU; every pair run so far
# run resumes: (cell, condition) pairs already in gens.jsonl are skipped.
import argparse
import csv
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

PAIRS = {  # base key, instruct key (sweep.MODELS), FLORES-selected head
    "qwen2.5-1.5b": ("qwen-base", "qwen-instruct", "L22H6"),
    "qwen2.5-3b": ("qwen2.5-3b", "qwen2.5-3b-instruct", "L27H13"),
    "qwen2.5-7b": ("qwen2.5-7b", "qwen2.5-7b-instruct", "L19H1"),
    "qwen3-1.7b": ("qwen3-1.7b", "qwen3-1.7b-instruct", "L18H12"),
    "gemma3-1b": ("gemma3-1b", "gemma3-1b-instruct", "L11H3"),
    "gemma3-4b": ("gemma3-4b", "gemma3-4b-instruct", "L24H0"),
    "olmo2-1b": ("olmo2-1b", "olmo2-1b-instruct", "L12H8"),
}
QWEN25_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def formats(pair):
    return ["raw", "chat", "chat+system"] if pair.startswith("qwen2.5") else ["raw", "chat"]


def parse(h):
    l, k = h[1:].split("H")
    return int(l), int(k)


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


def read_rows(path, per_lang, skip_screen):
    seen, rows = {}, []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        n = seen[r["language"]] = seen.get(r["language"], 0) + 1
        if skip_screen and n <= 25:  # the first 25 per language are the 125-prompt head screen
            continue
        if per_lang is None or n - (25 if skip_screen else 0) <= per_lang:
            rows.append(r)
    return rows


def render(itok, pair, prompt, fmt, template_date):
    if fmt == "raw":
        return prompt
    if fmt == "chat" and pair.startswith("qwen2.5"):
        # without a system turn, Qwen2.5's template inserts its default system prompt; write the turn by hand
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    msgs = [{"role": "user", "content": prompt}]
    if fmt == "chat+system":
        msgs = [{"role": "system", "content": QWEN25_SYSTEM}] + msgs
    text = itok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                    strftime_now=template_date.strftime)
    if itok.bos_token and itok("a").input_ids[0] == itok.bos_token_id:  # the tokenizer adds BOS itself
        text = text.removeprefix(itok.bos_token)
    return text


def out_dir(a):
    return Path(a.out or "out/format-grid")


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sweep import MODELS, NO_EOS, TEMPLATE_DATE, ablated, generate, load_kwargs

    base_key, inst_key, head = PAIRS[a.pair]
    rows = read_rows(a.prompts, a.per_lang, a.skip_screen)
    out = out_dir(a) / a.pair
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    itok = AutoTokenizer.from_pretrained(MODELS[inst_key][0])
    itok.pad_token = itok.pad_token or itok.eos_token
    btok = AutoTokenizer.from_pretrained(MODELS[base_key][0])
    probe = [r["prompt"] for r in rows[:: max(1, len(rows) // 50)]]
    same_ids = all(itok(p).input_ids == btok(p).input_ids for p in probe)
    if not same_ids:
        raise SystemExit(f"{a.pair}: base and instruct tokenizers give different ids on raw text; not running")
    texts = {fmt: [render(itok, a.pair, r["prompt"], fmt, TEMPLATE_DATE) for r in rows] for fmt in formats(a.pair)}
    if "system" in texts["chat"][0].replace(rows[0]["prompt"], "").lower():
        print(f"warning: {a.pair} chat template still contains a system turn", flush=True)
    dtype = a.dtype or MODELS[inst_key][1]
    l, h = parse(head)
    rng = random.Random(0)
    ctrl = []
    done = set()
    if (out / "gens.jsonl").exists():
        for line in open(out / "gens.jsonl", encoding="utf-8"):
            r = json.loads(line)
            done.add((r["cell"], r["cond"]))
    meta = {"pair": a.pair, "head": head, "dtype": dtype, "n": len(rows), "prompts": a.prompts,
            "per_lang": a.per_lang, "skip_screen": a.skip_screen, "same_ids": same_ids, "bs": a.bs,
            "chat_example": texts["chat"][0], "formats": formats(a.pair)}
    with open(out / "gens.jsonl", "a", encoding="utf-8") as f:
        for ckpt, key in (("base", base_key), ("instruct", inst_key)):
            model = None
            for fmt in formats(a.pair):
                cell = f"{ckpt}|{fmt}"
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(MODELS[key][0], dtype=getattr(torch, dtype),
                                                                 **load_kwargs(key)).cuda().eval()
                    cfg = model.config.get_text_config()
                    H = cfg.num_attention_heads
                    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
                    assert max(max(itok(t).input_ids) for t in texts[fmt][:50]) < model.get_input_embeddings().num_embeddings
                    if not ctrl and a.controls:
                        ctrl = sorted(rng.sample([k for k in range(H) if k != h], a.controls))
                        meta["controls"] = [f"L{l}H{k}" for k in ctrl]
                conds = [("none", None), (head, h)] + [(f"L{l}H{k}", k) for k in ctrl]
                prompts = texts[fmt]
                order = sorted(range(len(prompts)), key=lambda i: len(itok(prompts[i]).input_ids))
                min_new = 40 if key in NO_EOS else 0
                for cond, k in conds:
                    if (cell, cond) in done:
                        continue
                    t0 = time.time()
                    if k is None:
                        gen = generate(model, itok, prompts, order, a.bs, 40, min_new)
                    else:
                        with ablated(model, key, "head", l, k, dh):
                            gen = generate(model, itok, prompts, order, a.bs, 40, min_new)
                    f.write(json.dumps({"cell": cell, "cond": cond, "texts": gen}, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{a.pair} {cell:22s} {cond:8s} {time.time() - t0:6.1f}s", flush=True)
            del model
            torch.cuda.empty_cache()
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    label(out)


def _init():
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0


def _label(text):
    from langdetect import LangDetectException, detect
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def label(out):
    runs = [json.loads(line) for line in open(out / "gens.jsonl", encoding="utf-8")]
    uniq = sorted({t for r in runs for t in r["texts"]})
    with Pool(4, initializer=_init) as pool:
        lab = dict(zip(uniq, pool.map(_label, uniq, chunksize=200)))
    json.dump({f"{r['cell']}:{r['cond']}": [lab[t] for t in r["texts"]] for r in runs},
              open(out / "labels.json", "w", encoding="utf-8"))


def report(a):
    root = out_dir(a)
    lines = ["# Checkpoint x input format, FLORES-selected heads zeroed", "",
             "Same prompts and token ids in every cell. keep: share of non-English prompts continued in the prompt's "
             "language without ablation. c->w: share of all prompts in the prompt's language without ablation and "
             "in another language with the head zeroed (Eq. 1). chat: user turn only, no system turn; chat+system: "
             "Qwen2.5's default English system prompt, the paper's setting. ctrl: largest c->w among the same-layer "
             "controls, if run.", "",
             "| pair | head | n | format | base keep | base c->w | instruct keep | instruct c->w | ctrl (base / instr) |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for pair in PAIRS:
        out = root / pair
        if not (out / "labels.json").exists():
            continue
        meta = json.load(open(out / "meta.json", encoding="utf-8"))
        if (out / "prompts.csv").exists():
            expected = [r["language"] for r in csv.DictReader(open(out / "prompts.csv", encoding="utf-8"))]
        else:  # results/ keeps no copy of the prompts; rebuild them from the file named in meta.json
            expected = [r["language"] for r in read_rows(meta["prompts"], meta["per_lang"], meta["skip_screen"])]
        assert len(expected) == meta["n"]
        lab = json.load(open(out / "labels.json", encoding="utf-8"))
        non_en = [i for i, e in enumerate(expected) if e != "en"]
        head = meta["head"]
        summary[pair] = {}
        for fmt in meta["formats"]:
            cells = {}
            for ckpt in ("base", "instruct"):
                base = lab.get(f"{ckpt}|{fmt}:none")
                if base is None:
                    continue
                ok = [same(x, e) for x, e in zip(base, expected)]

                def c2w(cond):
                    y = lab.get(f"{ckpt}|{fmt}:{cond}")
                    return None if y is None else mean(b and not same(x, e) for b, x, e in zip(ok, y, expected))
                ctrl = [c2w(c) for c in meta.get("controls", [])]
                cells[ckpt] = {"keep": mean(ok[i] for i in non_en), "c2w": c2w(head),
                               "ctrl": max(ctrl) if ctrl and None not in ctrl else None}
            summary[pair][fmt] = cells

            def cell(ckpt, k):
                v = cells.get(ckpt, {}).get(k)
                return "-" if v is None else f"{v:.3f}"
            lines.append(f"| {pair} | {head} | {meta['n']} | {fmt} | {cell('base', 'keep')} | {cell('base', 'c2w')} | "
                         f"{cell('instruct', 'keep')} | {cell('instruct', 'c2w')} | "
                         f"{cell('base', 'ctrl')} / {cell('instruct', 'ctrl')} |")
    json.dump(summary, open(root / "summary.json", "w", encoding="utf-8"), indent=1)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--pair", choices=list(PAIRS))
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--per-lang", type=int, default=None, help="first N prompts per language, for smoke tests")
    p.add_argument("--skip-screen", action="store_true", help="leave out the 125 head-screen prompts")
    p.add_argument("--controls", type=int, default=0, help="random same-layer heads zeroed in every cell")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--dtype", default=None, help="default: the instruct model's precision")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.step == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
