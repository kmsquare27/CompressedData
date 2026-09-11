"""Step 11 -- read-only thesis compression report over Steps 02/06/07/08/09/10.

Place in src/pipeline/11_compression_report.py and run from any directory:
    python src/pipeline/11_compression_report.py --source webcode2m
    python src/pipeline/11_compression_report.py --source webcode2m --csv-only
    python src/pipeline/11_compression_report.py --source webcode2m \
        --compare-tokenizers gpt2 bigcode/starcoder2-3b

Dependencies: numpy pandas matplotlib pillow; full mode also needs transformers
and the fast tokenizer the project already uses. No browser or GPU is required.
Full mode recounts original/final HTML, verifies the sha256 values Step 08
recorded, and allocates whole-document tokens to lexical source categories via
the tokenizer's character offsets, so categories sum exactly to the document.
CSV-only mode uses recorded counts and does NOT invent category token counts.
Every run gets a new directory. Existing pipeline files are never modified.

What this script refuses to do, deliberately:
  * fall back to an approximate tokenizer -- a headline counted with chars/4
    is not the number the gate produced, and the difference is invisible
  * report HTML-token reduction without the estimated sequence reduction
    beside it -- images dominate a screenshot-to-code example
  * sum L2 token counterfactuals and L3 character buckets into one pie
  * treat a pixel-identical original fallback as a compression success
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LEVELS = ["original", "l1a", "l1b", "l1", "l2", "l3"]
# "unattributed" is index 0 and is the DEFAULT label. Any character the parser
# never visits stays here instead of masquerading as markup. On a corpus this
# malformed (see the lxml round-trip finding) a silent default of "markup"
# would hide exactly the pages where the attribution cannot be trusted.
CATEGORIES = ["unattributed", "markup", "attributes", "inline_css", "css_blocks",
              "scripts", "comments", "text", "whitespace", "special_tokens"]


def flag(v):
    return str(v).strip().lower() in {"true", "1", "1.0", "yes"}


def number(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def present(v):
    return v is not None and str(v).strip().lower() not in {"", "nan", "none"}


def read_csv(path, required=False, keys=("page_id",)):
    if not path.exists():
        if required:
            raise ValueError(f"Missing {path}. Run Step 08 first or supply the matching archived paths.")
        return pd.DataFrame()
    try:
        d = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for k in keys:
        if k not in d:
            raise ValueError(f"{path.name}: required column {k!r} missing")
    if keys and (d.duplicated(list(keys)).any() or d[list(keys)].eq("").any().any()):
        raise ValueError(f"{path.name}: duplicate or blank keys {keys}; resolve mixed runs first")
    return d


def indexed(d):
    return {r["page_id"]: r for r in d.to_dict("records")} if len(d) else {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value, root):
    if not present(value):
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    # Relative Windows paths also work when moved to a Unix machine.
    return root / str(value).replace("\\", "/")


def positive(v):
    return math.isfinite(number(v)) and number(v) > 0


def gini(x):
    """Concentration of savings across pages. 0 = every page contributes
    equally; 1 = one page carries the corpus. Negative savings are clipped
    (the Lorenz curve is undefined otherwise; G1 forbids them anyway)."""
    v = np.sort(np.clip(np.asarray(x, dtype=float), 0, None))
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.sum() == 0:
        return None
    return float((2.0 * np.arange(1, len(v) + 1) - len(v) - 1).dot(v) / (len(v) * v.sum()))


def image_tokens(width, height, factor=28, merge=2, min_pixels=56*56,
                 max_pixels=1280*28*28):
    """Qwen2-VL / 2.5-VL smart_resize, then patch count after the 2x2 merge.

    ESTIMATE, not the processor's output: it reproduces the documented
    resize rule but not your collator, chat template or prompt. It exists so
    the HTML-only reduction is never mistaken for a sequence reduction.
    Replace with processor-derived counts once the trainer exists (RQ3).
    """
    if width <= 0 or height <= 0:
        return 0
    rnd = lambda x: max(factor, int(round(x/factor))*factor)
    h, w = rnd(height), rnd(width)
    if h*w > max_pixels:
        beta = math.sqrt((height*width)/max_pixels)
        h = max(factor, math.floor(height/beta/factor)*factor)
        w = max(factor, math.floor(width/beta/factor)*factor)
    elif h*w < min_pixels:
        beta = math.sqrt(min_pixels/(height*width))
        h = math.ceil(height*beta/factor)*factor
        w = math.ceil(width*beta/factor)*factor
    return int((h//factor)*(w//factor)//(merge*merge))


def finite_only(obj):
    """json.dumps(allow_nan=False) raises on NaN, and default=str only
    handles unserializable TYPES. One NaN aggregate would kill a real run
    after the expensive part had already finished."""
    if isinstance(obj, dict):
        return {k: finite_only(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_only(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return finite_only(float(obj))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def stats(d, orig="tokens_orig", final="tokens_effective"):
    o = pd.to_numeric(d[orig], errors="coerce").to_numpy(float)
    f = pd.to_numeric(d[final], errors="coerce").to_numpy(float)
    mask = np.isfinite(o) & np.isfinite(f) & (o > 0) & (f >= 0)
    o, f = o[mask], f[mask]
    r = (o - f) / o * 100
    result = {"n_rows": len(d), "n_known": len(o), "n_unknown": int((~mask).sum()),
              "tokens_original": int(o.sum()), "tokens_final": int(f.sum()),
              "tokens_saved": int((o-f).sum())}
    result.update({"corpus_reduction_pct": float((o-f).sum()/o.sum()*100) if len(o) else None,
                   "compression_factor": float(o.sum()/f.sum()) if f.sum() > 0 else None,
                   "page_mean_reduction_pct": float(r.mean()) if len(r) else None,
                   "page_std_reduction_pct": float(r.std(ddof=1)) if len(r) > 1 else None})
    for q in [0, 10, 25, 50, 75, 90, 95, 100]:
        result[f"page_p{q}_reduction_pct"] = float(np.percentile(r, q)) if len(r) else None
    return result


def bootstrap(d, n, seed, cluster_col=None):
    """Resample whole pages or supplied site/template groups, keeping pairs together."""
    d = d[np.isfinite(d.tokens_orig) & np.isfinite(d.tokens_effective) & (d.tokens_orig > 0)]
    if n == 0 or len(d) < 2:
        return {"unit": cluster_col or "page", "n_resamples": 0, "interval": None}
    if cluster_col:
        if d[cluster_col].eq("").any():
            raise ValueError(f"Missing {cluster_col} for some pages; cannot perform group bootstrap")
        groups = [g[["tokens_orig", "tokens_effective"]].to_numpy(float)
                  for _, g in d.groupby(cluster_col, sort=True)]
    else:
        groups = [r.reshape(1, 2) for r in d[["tokens_orig", "tokens_effective"]].to_numpy(float)]
    if len(groups) < 2:
        return {"unit": cluster_col or "page", "n_groups": len(groups), "n_resamples": 0, "interval": None}
    # Aggregate first, so 10k pages x 2k bootstrap runs do not duplicate HTML/dataframes.
    agg = np.array([[g[:, 0].sum(), g[:, 1].sum(), len(g),
                     ((g[:, 0]-g[:, 1])/g[:, 0]*100).sum()] for g in groups])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        a = agg[rng.integers(0, len(agg), len(agg))].sum(axis=0)
        vals.append([(a[0]-a[1])/a[0]*100, a[3]/a[2]])
    limits = np.percentile(np.asarray(vals), [2.5, 97.5], axis=0)
    return {"unit": cluster_col or "page", "n_groups": len(groups), "n_resamples": n,
            "corpus_reduction_pct_95ci": limits[:, 0].tolist(),
            "page_mean_reduction_pct_95ci": limits[:, 1].tolist()}


class SourceMap(HTMLParser):
    """Lexical source accounting, NOT a DOM visibility or causal edit analysis.

    Assign each character once. CSS/script bodies keep their own whitespace.
    Attributes include their name/value, while tag delimiters remain markup.
    Source text can be hidden: only a renderer can determine visible text.
    """
    ATTR = re.compile(r'''\s+([^\s=/>]+)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?''')

    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text = text
        self.labels = np.zeros(len(text), dtype=np.uint8)
        self.lines = [0] + [m.end() for m in re.finditer("\n", text)]
        self.context = None
        self.feed(text)
        self.close()

    def pos(self):
        line, col = self.getpos()
        return self.lines[line-1] + col

    def mark(self, a, b, name):
        self.labels[a:b] = CATEGORIES.index(name)

    def handle_starttag(self, tag, attrs):
        raw, pos = self.get_starttag_text(), self.pos()
        self.mark(pos, pos + len(raw), "markup")
        for m in self.ATTR.finditer(raw):
            name = "inline_css" if m.group(1).lower() == "style" else "attributes"
            self.mark(pos+m.start(), pos+m.end(), name)
        if tag in {"style", "script"}:
            self.context = tag

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.context = None

    def handle_endtag(self, tag):
        p = self.pos()
        m = re.match(r"</[^>]*>", self.text[p:])
        if m:
            self.mark(p, p + m.end(), "markup")
        if tag == self.context:
            self.context = None

    def handle_decl(self, decl):
        p = self.pos()
        m = re.match(r"<![^>]*>", self.text[p:])
        if m:
            self.mark(p, p + m.end(), "markup")

    def handle_pi(self, data):
        p = self.pos()
        self.mark(p, p + len(data) + 2, "markup")

    def handle_data(self, data):
        p = self.pos()
        if self.context:
            self.mark(p, p+len(data), "css_blocks" if self.context == "style" else "scripts")
        else:
            self.mark(p, p+len(data), "text")
            for m in re.finditer(r"\s+", data):
                self.mark(p+m.start(), p+m.end(), "whitespace")

    def handle_comment(self, data):
        start = self.pos()
        end = self.text.find("-->", start)
        self.mark(start, len(self.text) if end < 0 else end+3, "comments")

    def entity(self):
        p = self.pos()
        m = re.match(r"&(?:#x[0-9a-fA-F]+|#\d+|[a-zA-Z0-9]+);?", self.text[p:])
        if m:
            self.mark(p, p+m.end(), "text")

    def handle_entityref(self, name):
        self.entity()

    def handle_charref(self, name):
        self.entity()


def describe_html(text, tokenizer):
    mapping = SourceMap(text)
    chars = np.bincount(mapping.labels, minlength=len(CATEGORIES))
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True,
                    truncation=False, return_attention_mask=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    counts = np.zeros(len(CATEGORIES), dtype=int)
    for start, end in offsets:
        if end <= start:
            counts[-1] += 1
        else:
            # Each whole-document token is assigned to the midpoint of its
            # source span. Boundary allocation is conventional, not causal.
            mid = min(len(text)-1, (start+end-1)//2)
            counts[int(mapping.labels[mid])] += 1
    if int(counts.sum()) != len(ids) or int(chars.sum()) != len(text):
        raise ValueError("Source allocation failed to conserve totals")
    return len(ids), counts, chars


def load_tokenizer(args, env):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ValueError("Install transformers for full reporting, or use --csv-only for logged counts.") from e
    revision = args.revision
    if not revision and args.tokenizer == MODEL:
        old = str(env.get("tokenizer", ""))
        if "@" in old and re.fullmatch(r"[0-9a-f]{40}", old.split("@")[-1]):
            revision = old.split("@")[-1]
    tk = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision,
                                      use_fast=True, trust_remote_code=False,
                                      local_files_only=args.local_files_only)
    if not tk.is_fast:
        raise ValueError("A fast tokenizer with offsets is required for category accounting")
    return tk, {"model": args.tokenizer, "requested_revision": revision,
                "resolved_commit": tk.init_kwargs.get("_commit_hash"),
                "backend_sha256": hashlib.sha256(tk.backend_tokenizer.to_str().encode()).hexdigest(),
                "add_special_tokens": True,
                "note": "HTML target tokens only; training chat/image/padding tokens are not counted here."}



def top_decile_share(saved):
    v = np.sort(np.clip(np.asarray(saved, dtype=float), 0, None))[::-1]
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.sum() == 0:
        return None
    k = max(1, int(round(0.10*len(v))))
    return float(100*v[:k].sum()/v.sum())


def sequence_accounting(pages, root, max_pixels, notes):
    """HTML tokens are not the sequence. Images dominate a screenshot-to-code
    example, so a reduction stated only against HTML overstates the compute
    claim by the image share. This is an ESTIMATE from the harness renders
    (documented smart_resize rule); it excludes prompt/chat overhead and is
    superseded by processor-derived counts once the trainer exists."""
    try:
        from PIL import Image
    except ImportError:
        notes.append("Pillow unavailable; image-token estimate omitted. "
                     "HTML-token reductions are NOT sequence reductions.")
        return {}
    ok = pages[pages.validated & np.isfinite(pages.tokens_orig)
               & np.isfinite(pages.tokens_effective)]
    counts, missing = [], 0
    for row in ok.to_dict("records"):
        path = resolve_path(row.get("png_path"), root)
        if path is None or not path.is_file():
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                counts.append(image_tokens(im.width, im.height, max_pixels=max_pixels))
        except Exception:
            missing += 1
    if not counts:
        notes.append("No readable harness renders; image-token estimate omitted.")
        return {}
    if missing:
        notes.append(f"{missing} validated pages had no readable render; the "
                     f"image-token estimate covers the remaining {len(counts)}.")
    img = float(np.sum(counts))
    scale = len(counts)/len(ok)
    html_o = float(ok.tokens_orig.sum())*scale
    html_f = float(ok.tokens_effective.sum())*scale
    seq_o, seq_f = img + html_o, img + html_f
    notes.append("Sequence accounting is an ESTIMATE from render dimensions and "
                 "the documented resize rule; it excludes prompt/chat/padding "
                 "tokens. Report the HTML figure and the sequence figure "
                 "together, never the HTML figure alone as a compute claim.")
    return {"pages_measured": len(counts), "max_pixels": max_pixels,
            "image_tokens_total": img,
            "image_tokens_median": float(np.median(counts)),
            "image_share_of_sequence_pct": 100*img/seq_o if seq_o else None,
            "html_reduction_pct": 100*(html_o-html_f)/html_o if html_o else None,
            "sequence_reduction_pct": 100*(seq_o-seq_f)/seq_o if seq_o else None}


def tokenizer_sensitivity(pages, root, args, notes):
    """Does the headline survive a different vocabulary? Markup is exactly
    where BPE merges disagree between models. Corpus reduction only: these
    counts are NOT the project's counts and must never replace them."""
    if args.csv_only or not args.compare_tokenizers:
        return pd.DataFrame()
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return pd.DataFrame()
    ok = pages[pages.validated]
    pairs = []
    for row in ok.to_dict("records"):
        a = resolve_path(row.get("original_html_path"), root)
        b = resolve_path(row.get("final_html_path"), root)
        if a and b and a.is_file() and b.is_file():
            pairs.append((a.read_text(encoding="utf-8", errors="ignore"),
                          b.read_text(encoding="utf-8", errors="ignore")))
    if not pairs:
        return pd.DataFrame()
    rows = []
    for name in [args.tokenizer] + list(args.compare_tokenizers):
        try:
            tk = AutoTokenizer.from_pretrained(
                name, use_fast=True, trust_remote_code=False,
                local_files_only=args.local_files_only)
            o = sum(len(tk(x, add_special_tokens=False)["input_ids"]) for x, _ in pairs)
            f = sum(len(tk(y, add_special_tokens=False)["input_ids"]) for _, y in pairs)
        except Exception as e:
            notes.append(f"Tokenizer {name} unavailable for sensitivity: {e}")
            continue
        rows.append({"tokenizer": name, "is_reported": name == args.tokenizer,
                     "tokens_original": o, "tokens_final": f,
                     "corpus_reduction_pct": 100*(o-f)/o if o else None})
    if len(rows) > 1:
        notes.append("Tokenizer sensitivity counts exclude special tokens and "
                     "are for robustness only; the reported headline uses the "
                     "project tokenizer.")
    return pd.DataFrame(rows)


def stage_tables(report_dir, source, pages, notes):
    original = pages.set_index("page_id").tokens_orig_logged.to_dict()
    eligible = set(pages.page_id)
    rows = []
    for stage, name, key in [
        ("l1a", f"level1_stages_{source}.csv", ("page_id", "stage")),
        ("l1b", f"level1_stages_{source}.csv", ("page_id", "stage")),
        ("l1", f"level1_gate_{source}.csv", ("page_id",)),
        ("l2", f"level2_gate_{source}.csv", ("page_id",)),
        ("l3", f"level3_gate_{source}.csv", ("page_id",))]:
        d = read_csv(report_dir / name, keys=key)
        if d.empty:
            notes.append(f"Optional {name} unavailable/empty; {stage} history omitted.")
            continue
        if "stage" in key:
            d = d[d.stage == stage]
        for r in d.to_dict("records"):
            pid = r["page_id"]
            if pid not in eligible:
                continue
            base = number(r.get("tokens_base", r.get("tokens_orig")))
            comp = number(r.get("tokens_comp"))
            ok = flag(r.get("accepted"))
            o = original[pid]
            rows.append({"page_id": pid, "stage": stage, "status": r.get("status", ""),
                         "accepted": ok, "base": r.get("base", "original"),
                         "tokens_base_logged": base, "tokens_candidate_logged": comp,
                         "tokens_original_report": o,
                         "tokens_saved_vs_base_logged": base-comp if ok else 0,
                         "candidate_reduction_vs_original_pct": 100*(o-comp)/o if ok and positive(o) else np.nan,
                         "pixel_identical": flag(r.get("pixel_identical"))})
    detail = pd.DataFrame(rows)
    aggregate = []
    if len(detail):
        for stage, g in detail.groupby("stage", sort=False):
            a = g[g.accepted]
            aggregate.append({"stage": stage, "attempted": len(g), "accepted": len(a),
                              "acceptance_pct": 100*len(a)/len(g),
                              "accepted_pixel_identical": int(a.pixel_identical.sum()),
                              "accepted_local_tokens_saved_logged": a.tokens_saved_vs_base_logged.sum(min_count=1),
                              "mean_accepted_candidate_reduction_vs_original_pct": a.candidate_reduction_vs_original_pct.mean()})
    return detail, pd.DataFrame(aggregate)


def ancillary(report_dir, source, pages, out, notes):
    """Keep units and scopes explicit; never relabel characters as tokens."""
    ids = set(pages.page_id)
    l2 = read_csv(report_dir / f"level2_gate_{source}.csv")
    if len(l2):
        l2 = l2[l2.page_id.isin(ids)]
        if "accepted" in l2:
            l2 = l2[l2.accepted.map(flag)]
        cols = [c for c in ["page_id", "base", "n_kept_invisible", "n_kept_wrappers",
                "tokens_saved_invisible", "tokens_saved_wrappers", "gate_calls"] if c in l2]
        l2[cols].to_csv(out / "l2_operator_diagnostics.csv", index=False)
        notes.append("L2 operator token deltas are separate counterfactuals; they need not add to total L2 savings and include serialization interactions.")
    b = read_csv(report_dir / f"level3_buckets_{source}.csv", keys=("page_id", "bucket"))
    gate = read_csv(report_dir / f"level3_gate_{source}.csv")
    if len(b) and len(gate) and "accepted" in gate:
        acc_ids = set(gate.loc[gate.accepted.map(flag), "page_id"]) & ids
        b = b[b.page_id.isin(acc_ids)].copy()
        if "chars" in b and len(b):
            b["chars"] = pd.to_numeric(b.chars, errors="coerce")
            b.groupby("bucket").agg(characters_removed=("chars", "sum"),
                                    pages=("page_id", "nunique")).reset_index().to_csv(out / "l3_character_buckets.csv", index=False)
            notes.append("L3 buckets: characters removed in stage-accepted candidates, not token counts or an attribution to the final dataset.")
    census = read_csv(report_dir / f"css_census_buckets_{source}.csv", keys=("page_id", "bucket"))
    if len(census):
        census[census.page_id.isin(ids)].to_csv(out / "css_census_proposals.csv", index=False)
        notes.append("CSS census contains proposed opportunity on its recorded base, not realized final savings; individual token deltas are not additive.")


def figures(pages, stages, categories, budget, out, sequence=None, sensitivity=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    figdir = out / "figures"
    figdir.mkdir()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(figdir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    measured = pages[np.isfinite(pages.reduction_pct)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(measured.reduction_pct, bins=min(25, max(1, len(measured))), color="#167c80", edgecolor="white")
    ax.set(xlabel="Original-to-final token reduction (%)", ylabel="Pages",
           title="All input pages with known counts (excluded pages: zero saving)")
    save(fig, "01_reduction_distribution")

    ok = pages[pages.validated]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("tokens_orig", "Original"), ("tokens_effective", "Final")]:
        v = np.sort(ok[col].dropna().to_numpy(float))
        if len(v):
            ax.step(v, np.arange(1, len(v)+1)/len(v), where="post", label=label)
    ax.set(xlabel="HTML target tokens", ylabel="Fraction of validated pages",
           title="Token length distribution on the same validated pages")
    if len(ok):
        ax.legend()
    save(fig, "02_length_ecdf")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(measured.tokens_orig, measured.reduction_pct, s=20, alpha=.65, color="#167c80")
    ax.set(xlabel="Original HTML tokens", ylabel="Token reduction (%)", title="Does page size affect compression?")
    save(fig, "03_size_vs_reduction")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = pages.final_level.value_counts().reindex(LEVELS + ["excluded"], fill_value=0)
    counts = counts[counts > 0]
    ax.bar(counts.index, counts.values, color="#167c80")
    ax.set(ylabel="Pages", title="Final target selection after Step 08")
    for i, n in enumerate(counts.values):
        ax.text(i, n, str(n), ha="center", va="bottom")
    ax.margins(y=.16)
    save(fig, "04_final_level_mix")

    if len(stages):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(stages))
        ax.bar(x-.2, stages.attempted, width=.4, label="Attempted", color="#b6d7d8")
        ax.bar(x+.2, stages.accepted, width=.4, label="Stage accepted", color="#167c80")
        ax.set(xticks=x, xticklabels=stages.stage, ylabel="Pages",
               title="Stage acceptance (these are not final selections)")
        ax.legend()
        save(fig, "05_stage_acceptance")

    known = pages[np.isfinite(pages.tokens_orig) & np.isfinite(pages.tokens_effective)]
    if len(known):
        # Provisional best-available history: no unverified fallback allocation.
        names = [c for c in ["tokens_orig", "best_through_l1_logged", "best_through_l2_logged",
                             "best_through_l3_logged", "tokens_effective"] if c in known]
        vals = [known[c].sum() for c in names]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(len(vals)), vals, "o-", color="#167c80")
        labels = {"tokens_orig": "Original", "best_through_l1_logged": "Through L1*",
                  "best_through_l2_logged": "Through L2*", "best_through_l3_logged": "Through L3*",
                  "tokens_effective": "Final"}
        ax.set(xticks=range(len(vals)), xticklabels=[labels[c] for c in names],
               ylabel="Corpus target tokens", title="Cumulative availability; * stage accepted, provisional")
        save(fig, "06_cumulative_token_totals")

    transition = pd.crosstab(pages.selected_level, pages.final_level)
    if len(transition):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.imshow(transition.values, cmap="Blues", aspect="auto")
        ax.set(xticks=range(len(transition.columns)), xticklabels=transition.columns,
               yticks=range(len(transition.index)), yticklabels=transition.index,
               xlabel="Final level", ylabel="Step 07 selected level", title="Final validation and fallback")
        for i in range(len(transition.index)):
            for j in range(len(transition.columns)):
                ax.text(j, i, str(transition.iloc[i, j]), ha="center", va="center", color="#111111")
        save(fig, "07_selection_transition")

    if len(categories):
        totals = categories.groupby("category")[["tokens_original", "tokens_final"]].sum().reindex(CATEGORIES)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(totals))
        ax.barh(x-.2, totals.tokens_original, height=.4, label="Original", color="#b6d7d8")
        ax.barh(x+.2, totals.tokens_final, height=.4, label="Final", color="#167c80")
        ax.set(yticks=x, yticklabels=totals.index, xlabel="Whole-document tokens allocated by source span",
               title="Lexical token composition on validated pairs")
        ax.legend()
        save(fig, "08_token_composition")
        fig, ax = plt.subplots(figsize=(9, 5))
        delta = totals.tokens_original - totals.tokens_final
        ax.barh(delta.index, delta.values, color=["#167c80" if v >= 0 else "#be694a" for v in delta])
        ax.axvline(0, color="#777777", lw=.8)
        ax.set(xlabel="Net allocated tokens saved (negative = growth)", title="Source-category changes, not causal operator attribution")
        save(fig, "09_category_net_savings")

    if len(budget):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(budget.html_budget, budget.original_fit_pct, "o-", label="Original")
        ax.plot(budget.html_budget, budget.final_fit_pct, "s-", label="Final")
        ax.set(xscale="log", xlabel="HTML-only token budget", ylabel="Validated pairs fitting budget (%)",
               title="Target budget coverage (excludes image, prompt and chat overhead)")
        ax.legend()
        save(fig, "10_target_budget_coverage")


    if sequence:
        img = sequence["image_tokens_total"]
        html_o = img*(100/sequence["image_share_of_sequence_pct"] - 1) if sequence["image_share_of_sequence_pct"] else 0
        html_f = html_o*(1 - sequence["html_reduction_pct"]/100) if sequence["html_reduction_pct"] is not None else html_o
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.barh(["Original", "Compressed"], [img, img], color="#b6d7d8", label="Image tokens (estimated)")
        ax.barh(["Original", "Compressed"], [html_o, html_f], left=[img, img],
                color="#167c80", label="HTML target tokens")
        ax.set(xlabel="Corpus tokens",
               title=f"HTML reduction {sequence['html_reduction_pct']:.1f}% = "
                     f"{sequence['sequence_reduction_pct']:.1f}% of the estimated sequence")
        ax.legend(loc="lower right")
        save(fig, "12_sequence_composition")

    if sensitivity is not None and len(sensitivity) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#be694a" if r else "#167c80" for r in sensitivity.is_reported]
        ax.bar(range(len(sensitivity)), sensitivity.corpus_reduction_pct, color=colors)
        ax.set(xticks=range(len(sensitivity)),
               xticklabels=[t.split("/")[-1] for t in sensitivity.tokenizer],
               ylabel="Corpus reduction (%)",
               title="Headline under other vocabularies (orange = reported)")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        save(fig, "13_tokenizer_sensitivity")

    p = out / "l3_character_buckets.csv"
    if p.exists():
        b = pd.read_csv(p).sort_values("characters_removed")
        fig, ax = plt.subplots(figsize=(9, max(4, .32*len(b))))
        ax.barh(b.bucket, b.characters_removed, color="#167c80")
        ax.set(xlabel="Characters removed", title="L3 operator buckets on stage-accepted pages")
        save(fig, "11_l3_character_buckets")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--root", type=Path, help="Project root, containing data/ and reports/")
    ap.add_argument("--manifest", type=Path, help="Step 08 compressed manifest")
    ap.add_argument("--final-report", type=Path, help="Step 08 final-validation CSV")
    ap.add_argument("--original-manifest", type=Path, help="Matching pilot/original manifest")
    ap.add_argument("--selection-report", type=Path, help="Matching Step 07 level-selection CSV")
    ap.add_argument("--report-dir", type=Path, help="Directory containing stage/census CSVs")
    ap.add_argument("--out", type=Path, help="NEW output directory; existing directories are refused")
    ap.add_argument("--csv-only", action="store_true", help="Use saved counts; skip tokenizer and artifact verification")
    ap.add_argument("--tokenizer", default=MODEL)
    ap.add_argument("--revision", help="Pin tokenizer revision; otherwise reuse Step 08 recorded SHA if available")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cluster-column", help="Site/template group column from the original or compressed manifest")
    ap.add_argument("--budgets", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--max-pixels", type=int, default=1280*28*28,
                    help="vision-encoder pixel cap used in training; sets the image-token estimate")
    ap.add_argument("--compare-tokenizers", nargs="*", default=[],
                    help="extra tokenizers for a sensitivity table (corpus reduction only)")
    args = ap.parse_args()
    if args.bootstrap < 0 or any(v <= 0 for v in args.budgets):
        ap.error("Bootstrap must be nonnegative and budgets must be positive")
    roots = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    root = args.root.resolve() if args.root else next((p for p in roots if (p/"data"/"splits").is_dir() and (p/"reports").is_dir()), None)
    if root is None:
        ap.error("Cannot locate project root. Put file 11 in src/pipeline or pass --root PATH")
    rep = (args.report_dir or root/"reports"/"csv").resolve()
    paths = {"final": args.final_report or rep/f"final_validation_{args.source}.csv",
             "manifest": args.manifest or root/"data"/"splits"/f"compressed_{args.source}_manifest.csv",
             "original": args.original_manifest or root/"data"/"splits"/f"pilot_{args.source}_manifest.csv",
             "selection": args.selection_report or rep/f"level_selection_{args.source}.csv",
             "environment": rep/f"final_validation_{args.source}_environment.json"}
    f = read_csv(paths["final"], required=True)
    m = read_csv(paths["manifest"], required=True)
    original = read_csv(paths["original"])
    selected = read_csv(paths["selection"])
    if f.empty or "status" not in f:
        raise ValueError("Final-validation report is empty or lacks status")
    mi, oi, si = indexed(m), indexed(original), indexed(selected)
    if set(f.loc[f.status == "ok", "page_id"]) != set(mi):
        raise ValueError("Step 08 successful IDs and compressed manifest IDs disagree. Use the same run for both.")
    notes = ["Headline cohort is ALL rows of Step 08, not only pages whose compression succeeded.",
             "Failed/unstable pages receive zero saving for corpus accounting only; they are not training examples.",
             "Original fallbacks can be pixel-identical without any compression; both rates are reported separately."]
    env = json.loads(paths["environment"].read_text()) if paths["environment"].exists() else {}
    tk, tokenizer_meta = (None, {"mode": "logged_csv", "recorded_tokenizer": env.get("tokenizer")}) if args.csv_only else load_tokenizer(args, env)
    if args.csv_only:
        notes.append("CSV-only mode: HTML bytes/hashes and token recounts were not verified; category-token figures are omitted.")
    if args.tokenizer != MODEL:
        notes.append("Alternative tokenizer requested: stage history uses logged project counts and is omitted from cumulative comparisons.")
    rows, cats, violations = [], [], []

    def check(condition, message):
        """Collected, not raised. A tokenizer load plus a full recount is a
        multi-minute round trip; surfacing one bad page at a time makes
        fixing a mixed run an afternoon."""
        if not condition:
            violations.append(message)
        return condition

    for i, r in enumerate(f.to_dict("records")):
        pid, valid = r["page_id"], r["status"] == "ok"
        mr, sr, op = mi.get(pid, {}), si.get(pid, {}), oi.get(pid, {})
        o, final = number(r.get("tokens_orig")), number(r.get("tokens_final"))
        if valid:
            for key in ("tokens_orig", "tokens_final"):
                if present(mr.get(key)):
                    check(number(mr[key]) == number(r.get(key)),
                          f"{pid}: {key} disagrees between final manifest and report")
            check(mr.get("level") == r.get("final_level"),
                  f"{pid}: final level differs between manifest and report")
            for key in ("original_sha256", "final_sha256"):
                if present(mr.get(key)) and present(r.get(key)):
                    check(mr[key] == r[key],
                          f"{pid}: {key} differs between manifest and report")
            check(positive(o) and math.isfinite(final) and final >= 0,
                  f"{pid}: invalid successful token counts")
        info = {"page_id": pid, "status": r["status"], "validated": valid,
                "selected_level": r.get("selected_level", "unknown"),
                "final_level": r.get("final_level", "original") if valid else "excluded",
                "final_pixel_identical": flag(r.get("final_pixel_identical")),
                "repeat_measured": present(r.get("original_repeat_identical")),
                "original_repeat_identical": flag(r.get("original_repeat_identical")),
                "rungs_tried": number(r.get("rungs_tried")),
                "tokens_orig_logged": o, "tokens_final_logged": final,
                "png_path": r.get("png_path") or op.get("png_path", ""),
                "artifact_hashes_verified": False}
        if args.cluster_column:
            info[args.cluster_column] = op.get(args.cluster_column) or mr.get(args.cluster_column, "")
        if tk is not None:
            raw_path = resolve_path(op.get("html_path"), root)
            if (raw_path is None or not raw_path.exists()) and present(sr.get("ladder")):
                ladder = json.loads(sr["ladder"])
                original_rungs = [a for a in ladder if a.get("level") == "original"]
                if original_rungs:
                    raw_path = resolve_path(original_rungs[-1].get("html_path"), root)
            if raw_path is None or not raw_path.is_file():
                raise ValueError(f"{pid}: cannot locate original HTML. Supply its matching archived original/selection manifest, or use --csv-only.")
            target = resolve_path(mr.get("html_path"), root) if valid else raw_path
            if target is None or not target.is_file():
                raise ValueError(f"{pid}: final HTML missing: {target}")
            verified = 0
            for p, expected in [(raw_path, r.get("original_sha256") or mr.get("original_sha256")),
                                (target, (r.get("final_sha256") or mr.get("final_sha256")) if valid else None)]:
                if present(expected):
                    if check(sha(p) == expected,
                             f"{pid}: file changed after Step 08: {p}"):
                        verified += 1
            info["artifact_hashes_verified"] = valid and verified == 2
            raw_bytes, final_bytes = raw_path.read_bytes(), target.read_bytes()
            # Match gate.py read_text's universal-newline behavior exactly.
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            final_text = target.read_text(encoding="utf-8", errors="ignore")
            o, oc, och = describe_html(raw_text, tk)
            final, fc, fch = describe_html(final_text, tk) if valid else (o, oc, och)
            info.update(original_html_path=str(raw_path), final_html_path=str(target),
                        bytes_original=len(raw_bytes), bytes_final=len(final_bytes),
                        token_recount_mismatch=(o != info["tokens_orig_logged"] or
                                                (valid and final != info["tokens_final_logged"])))
            if valid:
                for j, cat in enumerate(CATEGORIES):
                    cats.append({"page_id": pid, "category": cat, "tokens_original": int(oc[j]),
                                 "tokens_final": int(fc[j]), "tokens_saved": int(oc[j]-fc[j]),
                                 "characters_original": int(och[j]), "characters_final": int(fch[j])})
        effective = final if valid else o
        info.update(tokens_orig=o, tokens_effective=effective,
                    reduction_pct=100*(o-effective)/o if positive(o) and math.isfinite(effective) else np.nan,
                    actually_shorter=valid and effective < o)
        rows.append(info)
        if (i+1) % 100 == 0:
            print(f"[11] processed {i+1}/{len(f)} pages", flush=True)
    if violations:
        head = "\n  ".join(violations[:40])
        more = f"\n  ... and {len(violations)-40} more" if len(violations) > 40 else ""
        raise ValueError(f"{len(violations)} integrity violation(s); the run is a "
                         f"mixture of pipeline states and no number from it is "
                         f"safe to report:\n  {head}{more}")
    pages, categories = pd.DataFrame(rows), pd.DataFrame(cats)
    if tk is not None and pages.token_recount_mismatch.any():
        notes.append(f"{int(pages.token_recount_mismatch.sum())} page token recounts differ from logged values. Headline uses recounted values; cumulative stage history is omitted to avoid mixing tokenizers/counting modes.")
    history, stages = stage_tables(rep, args.source, pages, notes)
    same_counts = args.tokenizer == MODEL and (tk is None or not pages.token_recount_mismatch.any())
    if len(history) and same_counts:
        best = pages.set_index("page_id").tokens_orig.copy()
        for label, candidates in [("l1", ["l1a", "l1b", "l1"]), ("l2", ["l2"]), ("l3", ["l3"])]:
            for r in history[history.stage.isin(candidates) & history.accepted].to_dict("records"):
                v = r["tokens_candidate_logged"]
                if math.isfinite(v) and positive(best[r["page_id"]]):
                    best[r["page_id"]] = min(best[r["page_id"]], v)
            pages[f"best_through_{label}_logged"] = pages.page_id.map(best)
        notes.append("Cumulative stage curve is the best available stage-accepted target at each level, before final revalidation. It is not an independently revalidated L1/L2 ablation or a causal contribution waterfall.")
    out = (args.out or root/"reports"/"compression"/args.source/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pages.to_csv(out/"per_page.csv", index=False)
    history.to_csv(out/"stage_per_page.csv", index=False)
    stages.to_csv(out/"stage_summary.csv", index=False)
    if len(categories):
        categories.to_csv(out/"token_categories_per_page.csv", index=False)
        categories.groupby("category").sum(numeric_only=True).reindex(CATEGORIES).to_csv(out/"token_categories_summary.csv")
        notes.append("Source category counts allocate each whole-document token to its source-span midpoint; totals add exactly, but categories are lexical and not causal. A text node need not be visible; CSS/script whitespace remains inside CSS/script categories.")
    ok = pages[pages.validated]
    known_ok = ok[np.isfinite(ok.tokens_orig) & np.isfinite(ok.tokens_effective) & (ok.tokens_orig > 0)]
    budgets = []
    for b in sorted(set(args.budgets)):
        if len(known_ok):
            fits_o, fits_f = known_ok.tokens_orig <= b, known_ok.tokens_effective <= b
            budgets.append({"html_budget": b, "n_paired": len(known_ok), "original_fit_pct": fits_o.mean()*100,
                            "final_fit_pct": fits_f.mean()*100, "rescued_pages": int((~fits_o & fits_f).sum())})
    budget = pd.DataFrame(budgets)
    budget.to_csv(out/"target_budget_coverage.csv", index=False)
    groups = []
    for level, g in pages.groupby("final_level", sort=False):
        groups.append({"final_level": level, **stats(g)})
    pd.DataFrame(groups).to_csv(out/"final_level_summary.csv", index=False)
    sized = pages[np.isfinite(pages.tokens_orig) & (pages.tokens_orig > 0)].copy()
    if len(sized):
        sized["size_bucket"] = pd.qcut(sized.tokens_orig, 4, duplicates="drop")
        size_rows = [{"original_token_bucket": str(b), **stats(g)} for b, g in sized.groupby("size_bucket", observed=True)]
        pd.DataFrame(size_rows).to_csv(out/"size_bucket_summary.csv", index=False)
    sequence = sequence_accounting(pages, root, args.max_pixels, notes)
    if sequence:
        pd.DataFrame([sequence]).to_csv(out/"sequence_accounting.csv", index=False)
    sensitivity = tokenizer_sensitivity(pages, root, args, notes)
    if len(sensitivity):
        sensitivity.to_csv(out/"tokenizer_sensitivity.csv", index=False)
    ancillary(rep, args.source, pages, out, notes)
    summary = {"source": args.source, "created_utc": datetime.now(timezone.utc).isoformat(),
               "mode": "csv_only" if args.csv_only else "recount_and_verify", "tokenizer": tokenizer_meta,
               "all_input_accounting": stats(pages), "validated_training_pairs": stats(ok),
               "actually_shorter_validated_pairs": stats(ok[ok.actually_shorter]),
               "n_validated": len(ok), "n_excluded": len(pages)-len(ok),
               "n_shorter": int(ok.actually_shorter.sum()),
               "n_identical_validated": int(ok.final_pixel_identical.sum()),
               "n_shorter_and_identical": int((ok.actually_shorter & ok.final_pixel_identical).sum()),
               "n_original_fallbacks": int((ok.final_level == "original").sum()),
               "n_level_label_changes_after_selection": int((ok.selected_level != ok.final_level).sum()),
               "n_pages_trying_multiple_rungs": int((ok.rungs_tried > 1).sum()),
               "n_repeat_measured": int(pages.repeat_measured.sum()),
               "n_repeat_identical": int(pages.original_repeat_identical.sum()),
               "n_pairs_with_both_hashes_verified": int(ok.artifact_hashes_verified.sum()),
               "status_counts": pages.status.value_counts().to_dict(),
               "bootstrap_all_input_accounting": bootstrap(pages, args.bootstrap, args.seed, args.cluster_column),
               "bootstrap_validated_pairs": bootstrap(ok, args.bootstrap, args.seed, args.cluster_column),
               "savings_concentration": {
                   "gini": gini((known_ok.tokens_orig - known_ok.tokens_effective).to_numpy()),
                   "top_decile_share_pct": top_decile_share(
                       (known_ok.tokens_orig - known_ok.tokens_effective).to_numpy())},
               "sequence_accounting": sequence,
               "tokenizer_sensitivity": sensitivity.to_dict("records") if len(sensitivity) else [],
               "step08_environment": env, "notes": notes}
    snap = out/"input_snapshot"
    snap.mkdir()
    provenance = {}
    all_inputs = list(paths.values()) + list(rep.glob(f"*{args.source}*.csv"))
    for p in dict.fromkeys(all_inputs):
        if p.is_file():
            name = f"{len(provenance):02d}_{p.name}"
            shutil.copy2(p, snap/name)
            provenance[str(p.resolve())] = {"sha256": sha(p), "snapshot": name}
    summary["inputs"] = provenance
    summary["script_sha256"] = sha(Path(__file__))
    summary["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out/"summary.json").write_text(
        json.dumps(finite_only(summary), indent=2, allow_nan=False, default=str),
        encoding="utf-8")
    figures(pages, stages, categories, budget, out, sequence, sensitivity)
    a, v = summary["all_input_accounting"], summary["validated_training_pairs"]
    def fmt(x):
        return "unknown" if x is None else f"{x:.2f}%"
    text = ["# Compression report", "", f"Mode: {summary['mode']}. Source: {args.source}.", "",
            f"Input pages: {len(pages)}. Validated: {len(ok)}. Excluded: {len(pages)-len(ok)}.",
            f"Actually shorter validated targets: {summary['n_shorter']}; shorter AND pixel-identical: {summary['n_shorter_and_identical']}.",
            f"Corpus token reduction, all known inputs: **{fmt(a['corpus_reduction_pct'])}** ({a['n_unknown']} unknown denominators).",
            f"Corpus token reduction, validated pairs: **{fmt(v['corpus_reduction_pct'])}**.",
            f"Page-average reduction, all known inputs: {fmt(a['page_mean_reduction_pct'])}.", "",
            "Confidence intervals and full counts are in summary.json; source rows are in per_page.csv.", "",
            "## Interpretation", "", *[f"- {n}" for n in notes], "",
            "## Figures", ""]
    for p in sorted((out/"figures").glob("*.png")):
        text.extend([f"![{p.stem}](figures/{p.name})", ""])
    (out/"REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(f"[11] inputs={len(pages)} validated={len(ok)} shorter={summary['n_shorter']}")
    print(f"[11] corpus reduction={fmt(a['corpus_reduction_pct'])}; unknown denominators={a['n_unknown']}")
    print(f"[11] report: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as e:
        raise SystemExit(f"[11] ERROR: {e}") from e