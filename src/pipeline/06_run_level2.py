"""Step 06 -- Level 2: hierarchical DOM condensation, gate-validated.

Two subtractive transforms, applied EXHAUSTIVELY (the stress test applies the
same two once each, as safety probes -- this is the product):

  R1  remove invisible subtree roots   (display:none, and zero-ink subtrees
                                        with no paintable descendant)
  R2  collapse computed-neutral div/span wrappers

Both predicates are the ones the stress test validated (s5/s4 in
mutations.py), evaluated on BROWSER-COMPUTED style rather than markup, so a
stylesheet-hidden subtree or a stylesheet-neutral wrapper is found.

WHY BISECTION IS NOT OPTIONAL. The stress run measured that ~30% of
individual neutral-wrapper collapses change pixels (29/96 pages), the
margin-collapse trap: removing a wrapper can change whether adjacent margins
collapse, which no computed-style rule can predict. A page with ten
collapsible wrappers therefore almost never passes as a batch, and a
whole-page gate would reject it wholesale, discarding nine good edits to
punish one. So on rejection we bisect (delta-debugging-lite): split the edit
list, test the halves, recurse, blacklist the minimal offending subset and
keep the rest. Cost is ~O(k log n) renders instead of O(n); yield goes from
"nothing" to "everything except the few bad edits".

Every set reported as accepted has been gate-verified AS A WHOLE -- the
runner tracks the largest verified-passing set rather than assuming that two
independently-safe halves are safe together (they need not be: interaction
between edits is exactly what margin collapsing produces).

Composition: reads the Level 1 output when present, else the original, so
levels compose L1 -> L2 as the protocol specifies. Per-page level selection
happens downstream; this step just records what L2 achieved.

Usage:
    python src/pipeline/06_run_level2.py --source webcode2m --k 100

Output: outputs/level2/<source>/<id>.html           accepted candidates
        reports/csv/level2_gate_<source>.csv        per-page results
        reports/csv/level2_edits_<source>.csv       per-edit accept/blacklist
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import (TokenCounter, evaluate_pair, load_config,  # noqa: E402
                              render_page)
from src.compress.level1_minify import HTML_PARSER  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402
from src.stress import annotate  # noqa: E402
from src.stress.mutations import parse_css_color  # noqa: E402
from src.stress.annotate import STAMP_ATTR  # noqa: E402

NEVER_TOUCH = {"HTML", "BODY", "HEAD", "SCRIPT", "STYLE", "TITLE", "META",
               # zero-area by geometry but load-bearing by layout: a <br>
               # breaks a line, <col> sizes a column, <option>/<source>/
               # <track>/<param> belong to an atomic parent. Every one used
               # to be proposed as "invisible", rejected by the render gate,
               # and paid for in bisection budget.
               "BR", "WBR", "COL", "COLGROUP", "OPTION", "OPTGROUP",
               "SOURCE", "TRACK", "PARAM"}


# ---------------------------------------------------------------------------
# Candidate discovery (browser-computed, not markup-guessed)
# ---------------------------------------------------------------------------
def invisible_roots(meta) -> list[int]:
    """Subtree roots that put NO ink on the canvas.

    Two families: computed display:none (the s5 rule, validated), and
    elements that are themselves unpaintable AND have no paintable
    descendant. The second catches zero-area and parked-off-canvas wrappers
    whose children are also invisible -- but only with the descendant check,
    because a zero-height element with overflow:visible can still show its
    children, and deleting it would be a real breakage.

    Only ROOTS are returned: if an ancestor is already going, its descendants
    are redundant edits that would inflate the edit count and waste bisection
    budget on nothing.
    """
    kids: dict[int, list[int]] = {}
    for e in meta.els.values():
        kids.setdefault(e["parentId"], []).append(e["id"])

    memo: dict[int, bool] = {}

    def has_paintable(eid: int) -> bool:
        if eid in memo:
            return memo[eid]
        e = meta.els.get(eid)
        if e is None:
            return False
        memo[eid] = True  # cycle guard; DOM is a tree, this is belt-and-braces
        out = bool(e["paintable"]) or any(has_paintable(c)
                                          for c in kids.get(eid, []))
        memo[eid] = out
        return out

    dead = set()
    for e in meta.in_doc_order():
        if e["tag"] in NEVER_TOUCH or e.get("inSvg"):
            # SVG interiors: <defs>, <clipPath>, <linearGradient> are
            # zero-area yet referenced by url(#id); the <svg> root itself is
            # atomic and still a candidate when display:none.
            continue
        gone = e["displayNone"] or not has_paintable(e["id"])
        if gone:
            dead.add(e["id"])

    roots = []
    for eid in sorted(dead):
        p = meta.els.get(eid, {}).get("parentId", -1)
        seen = set()
        while p != -1 and p not in seen:      # any dead ancestor -> not a root
            seen.add(p)
            if p in dead:
                break
            p = meta.els.get(p, {}).get("parentId", -1)
        else:
            roots.append(eid)
            continue
        if p == -1 or p not in dead:
            roots.append(eid)
    return roots


def neutral_wrappers(meta, skip: set[int]) -> list[int]:
    """div/span whose COMPUTED style is provably layout-neutral -- the exact
    s4 predicate the stress test exercised. Returned DEEPEST-FIRST so nested
    wrappers collapse bottom-up and each unwrap sees a stable parent."""
    def neutral(e) -> bool:
        bg = parse_css_color(e["bg"])
        want_display = "block" if e["tag"] == "DIV" else "inline"
        return (e["tag"] in ("DIV", "SPAN") and not e["displayNone"]
                and not e.get("inSvg") and not e.get("wrapperHazard")
                and e["nChildNodes"] >= 1
                and e["leafTextLen"] == 0
                and sum(e["pad"]) == 0 and sum(e["mar"]) == 0
                and sum(e["borderW"]) == 0
                and (bg is None or bg[3] <= 0.01)
                and e["position"] == "static" and e["opacity"] >= 0.999
                and not e["transformed"] and e["overflowVisible"]
                and e["display"] == want_display
                and e["parentDisplay"] not in
                ("flex", "grid", "inline-flex", "inline-grid"))

    def depth(eid: int) -> int:
        d, p, seen = 0, meta.els.get(eid, {}).get("parentId", -1), set()
        while p != -1 and p not in seen:
            seen.add(p); d += 1
            p = meta.els.get(p, {}).get("parentId", -1)
        return d

    ids = [e["id"] for e in meta.in_doc_order()
           if e["id"] not in skip and neutral(e)]
    return sorted(ids, key=depth, reverse=True)


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------
def apply_edits(stamped_html: str, edits: list[tuple[str, int]]) -> str:
    """Apply an edit list to the stamped HTML and return UNSTAMPED output.

    Removals run before collapses so a collapse never targets a node that is
    about to disappear with its parent. Stamps are stripped last: they exist
    only to map browser ids onto soup nodes and must not reach the artifact
    (they would also inflate the token count they are meant to help reduce).
    """
    soup = BeautifulSoup(stamped_html, HTML_PARSER)      
    index = {}
    for el in soup.find_all(attrs={STAMP_ATTR: True}):
        index[int(el[STAMP_ATTR])] = el

    for kind, eid in [e for e in edits if e[0] == "remove"]:
        el = index.get(eid)
        if el is not None and el.parent is not None:
            el.decompose()
    for kind, eid in [e for e in edits if e[0] == "collapse"]:
        el = index.get(eid)
        if el is not None and el.parent is not None:
            el.unwrap()

    for el in soup.find_all(attrs={STAMP_ATTR: True}):
        del el[STAMP_ATTR]
    return str(soup)


# ---------------------------------------------------------------------------
# Gate-validated resolution with bisection
# ---------------------------------------------------------------------------
class Resolver:
    """Finds the largest gate-passing subset of a candidate edit list.

    Never reports a set it has not verified whole: `best` is only updated
    from an actual passing gate call, so edit interactions (margin collapse
    between two independently-safe removals) cannot slip through."""

    def __init__(self, page_id, stamped, art_base, cfg, harness, tk,
                 out_dir, work, budget):
        self.pid, self.stamped, self.art_base = page_id, stamped, art_base
        self.cfg, self.h, self.tk = cfg, harness, tk
        self.out_dir, self.work, self.budget = out_dir, work, budget
        self.calls = 0
        self.best: list[tuple[str, int]] = []
        self.best_R: dict | None = None
        self.blacklist: set[tuple[str, int]] = set()

    def gate(self, edits: list[tuple[str, int]]):
        """One render + gate evaluation. Returns (passed, R) or (None, None)
        when the per-page render budget is exhausted."""
        if self.calls >= self.budget:
            return None, None
        self.calls += 1
        html = apply_edits(self.stamped, edits)
        p = self.out_dir / f"{self.pid}__cand.html"
        p.write_text(html, encoding="utf-8")
        art = render_page(self.h, p, self.work / f"{self.pid}__cand.png", self.tk)
        R = evaluate_pair(self.art_base, art, self.cfg, page_id=self.pid)
        ok = bool(R["accepted"])
        # The objective is tokens, not edit count: ten wrapper collapses can
        # be worth less than one hidden-menu removal.
        if ok and (self.best_R is None or R["tokens_comp"] < self.best_R["tokens_comp"]):
            self.best, self.best_R = list(edits), R
        return ok, R

    def resolve(self, edits: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Return a subset of `edits` believed safe. Recursion is standard
        delta debugging: a failing set is split, each half resolved, and the
        union re-tested because two safe halves need not be safe together."""
        if not edits:
            return []
        ok, _ = self.gate(edits)
        if ok:
            return edits
        if ok is None:                       # budget spent
            return []
        if len(edits) == 1:
            self.blacklist.add(edits[0])
            return []
        mid = len(edits) // 2
        left = self.resolve(edits[:mid])
        right = self.resolve(edits[mid:])
        if left and right:
            ok2, _ = self.gate(left + right)
            if ok2:
                return left + right
            # Interaction between the halves: keep the bigger, which is
            # already verified on its own.
            return left if len(left) >= len(right) else right
        return left + right


# ---------------------------------------------------------------------------
# Per-source driver
# ---------------------------------------------------------------------------
def run_source(source, args, harness, cfg, tk):
    manifest = ROOT / "data" / "splits" / f"pilot_{source}_manifest.csv"
    if not manifest.exists():
        print(f"[06] no manifest at {manifest}"); return None
    df = pd.read_csv(manifest).head(args.k)

    l1_dir = ROOT / "outputs" / "level1" / source
    # Step 02 writes outputs/level1/<id>.html for EVERY page, before gating --
    # so the file existing proves nothing. Composing on a gate-REJECTED L1
    # output would stack L2 on a transformation already known to break the
    # page, and worse, the gate would then compare L2 against that broken
    # baseline and happily accept "looks like the broken version". Only pages
    # L1 actually passed are safe to build on; the rest restart from the
    # original, which is also what makes the "L1 rejected, L2 accepted" case
    # meaningful.
    l1_ok: set[str] = set()
    l1_csv = ROOT / "reports" / "csv" / f"level1_gate_{source}.csv"
    if l1_csv.exists():
        d1 = pd.read_csv(l1_csv)
        if "accepted" in d1.columns:
            l1_ok = set(d1.loc[d1["accepted"].astype(str).str.lower()
                               .isin(["true", "1"]), "page_id"].astype(str))
        print(f"[06] composing on Level 1 for {len(l1_ok)} gate-accepted pages; "
              f"the rest restart from the original")
    elif args.on_level1:
        print(f"[06] no {l1_csv.name}; running on originals "
              f"(run step 02 first to compose L1 -> L2)")

    out_dir = ROOT / "outputs" / "level2" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / source / "l2"
    work.mkdir(parents=True, exist_ok=True)
    rep = ROOT / "reports" / "csv"; rep.mkdir(parents=True, exist_ok=True)

    rows, edit_rows = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"level2[{source}]"):
        pid = str(r["page_id"])
        # Compose on Level 1's output when it exists; the protocol runs
        # L1 -> L2, each on the previous output.
        l1 = l1_dir / f"{pid}.html"
        use_l1 = args.on_level1 and pid in l1_ok and l1.exists()
        base_path = l1 if use_l1 else Path(r["html_path"])
        base_from = "level1" if use_l1 else "original"
        base_html = base_path.read_text(encoding="utf-8", errors="ignore")

        row = {"page_id": pid, "source": source, "base": base_from}
        if annotate.data_attr_selector_hazard(base_html):
            rows.append({**row, "status": "skipped_data_selector"}); continue
        stamped, _ = annotate.stamp_ids(base_html)
        if stamped is None:
            rows.append({**row, "status": "skipped_no_body"}); continue

        sp = out_dir / f"{pid}__stamped.html"
        sp.write_text(stamped, encoding="utf-8")
        try:
            # REFERENCE = render of the ORIGINAL file. Two reasons:
            #  (1) the stamped file is a BeautifulSoup round-trip of the base
            #      (str(soup)); validating against ITS render would validate
            #      against a rewritten page, not the original;
            #  (2) when composing on Level 1, gating against the original
            #      means L1's and L2's tolerances cannot accumulate -- the
            #      stacked artifact is held to ONE budget against the truth.
            # G1 (token reduction) is still measured against the base file
            # L2 edits, so `tokens_orig` keeps its "relative to the base"
            # semantics; `tokens_original` is added for step 07.
            art_ref = render_page(harness, Path(r["html_path"]),
                                  work / f"{pid}__orig.png", tk)
            tokens_original = art_ref.tokens
            art_ref.html_text = base_html
            art_ref.tokens = tk.count(base_html)
            art_st = render_page(harness, sp, work / f"{pid}__stamped.png", tk)
            meta = annotate.collect_meta(harness)
        except Exception as e:  # noqa: BLE001
            rows.append({**row, "status": f"page_error: {e}"}); continue
        art_base = art_ref
        row["stamp_roundtrip_identical"] = (
            bool(art_st.img.shape == art_ref.img.shape
                 and (art_st.img == art_ref.img).all())
            if base_from == "original" else None)

        removals = invisible_roots(meta)
        skip = set(removals)
        collapses = neutral_wrappers(meta, skip)
        edits = ([("remove", i) for i in removals]
                 + [("collapse", i) for i in collapses])
        row.update({"n_invisible": len(removals), "n_wrappers": len(collapses),
                    "n_candidate_edits": len(edits),
                    "tokens_base": art_base.tokens,
                    "tokens_original": tokens_original})
        if not edits:
            rows.append({**row, "status": "no_candidates", "accepted": False,
                         "reduction_pct": 0.0}); continue

        # PRESCREEN on the loaded stamped page: no renders. Only edits the
        # oracle cannot distinguish from a no-op reach the render gate; the
        # rest are recorded with the first differing property/rect. When the
        # oracle is unavailable the full list is gated as before.
        pre = annotate.prescreen_edits(harness, edits)
        pre_by = {(p["kind"], p["id"]): p for p in (pre or [])}
        if pre is not None:
            gated = [e for e in edits if pre_by.get(e, {}).get("ok")]
        else:
            gated = list(edits)
        row.update({"prescreen_ran": pre is not None,
                    "n_prescreen_dropped": len(edits) - len(gated)})
        if not gated:
            rows.append({**row, "status": "all_edits_rejected",
                         "accepted": False, "reduction_pct": 0.0,
                         "gate_calls": 0})
            for kind, eid in edits:
                edit_rows.append({"page_id": pid, "kind": kind, "el_id": eid,
                                  "prescreen_ok": False,
                                  "prescreen_first": pre_by.get((kind, eid), {}).get("first", ""),
                                  "kept": False, "blacklisted": False})
            continue
        edits = gated

        res = Resolver(pid, stamped, art_base, cfg, harness, tk,
                       out_dir, work, args.budget)
        try:
            res.resolve(edits)
        except Exception as e:  # noqa: BLE001
            rows.append({**row, "status": f"failed: {e}"}); continue

        kept, R = res.best, res.best_R
        if not kept or R is None:
            rows.append({**row, "status": "all_edits_rejected",
                         "accepted": False, "reduction_pct": 0.0,
                         "gate_calls": res.calls}); continue

        final_html = apply_edits(stamped, kept)
        (out_dir / f"{pid}.html").write_text(final_html, encoding="utf-8")

        # Attribute savings per rule (token counts only -- no extra renders).
        kr = [e for e in kept if e[0] == "remove"]
        kc = [e for e in kept if e[0] == "collapse"]
        t_rm = tk.count(apply_edits(stamped, kr)) if kr else art_base.tokens
        t_cl = tk.count(apply_edits(stamped, kc)) if kc else art_base.tokens
        rows.append({**row, "status": "ok", "gate_calls": res.calls,
                     "n_kept": len(kept), "n_kept_invisible": len(kr),
                     "n_kept_wrappers": len(kc),
                     "n_blacklisted": len(res.blacklist),
                     "tokens_saved_invisible": art_base.tokens - t_rm,
                     "tokens_saved_wrappers": art_base.tokens - t_cl,
                     **{k: v for k, v in R.items() if k != "page_id"}})
        for kind, eid in ([("remove", i) for i in removals]
                          + [("collapse", i) for i in collapses]):
            p = pre_by.get((kind, eid), {})
            edit_rows.append({"page_id": pid, "kind": kind, "el_id": eid,
                              "prescreen_ok": bool(p.get("ok", True)),
                              "prescreen_first": p.get("first", ""),
                              "kept": (kind, eid) in kept,
                              "blacklisted": (kind, eid) in res.blacklist})

        pd.DataFrame(rows).to_csv(rep / f"level2_gate_{source}.csv", index=False)
        pd.DataFrame(edit_rows).to_csv(rep / f"level2_edits_{source}.csv",
                                       index=False)

    pd.DataFrame(rows).to_csv(rep / f"level2_gate_{source}.csv", index=False)
    pd.DataFrame(edit_rows).to_csv(rep / f"level2_edits_{source}.csv", index=False)
    return rep / f"level2_gate_{source}.csv"


def summarize(csv_path, source):
    d = pd.read_csv(csv_path)
    ok = d[d["status"] == "ok"]
    print(f"\n[06] ---- Level 2 under the FROZEN gate: {source} "
          f"({len(d)} pages) ----")
    print(d["status"].value_counts().to_string())
    if not len(ok):
        print("[06] nothing accepted"); return
    acc = ok["accepted"].astype(bool)
    red = pd.to_numeric(ok.loc[acc, "reduction_pct"], errors="coerce")
    print(f"\naccepted: {int(acc.sum())}/{len(d)}")
    if len(red):
        print(f"reduction on accepted  mean {red.mean():.2f}%  "
              f"median {red.median():.2f}%  p90 {red.quantile(.9):.2f}%")
        print(f"corpus-level (rejected count as 0%): "
              f"{red.sum() / max(len(d), 1):.2f}%")
    inv = pd.to_numeric(ok.get("tokens_saved_invisible"), errors="coerce").fillna(0)
    wrp = pd.to_numeric(ok.get("tokens_saved_wrappers"), errors="coerce").fillna(0)
    tot = max(inv.sum() + wrp.sum(), 1)
    print(f"\nwhere the savings came from:")
    print(f"  invisible subtrees : {100 * inv.sum() / tot:5.1f}%  "
          f"({int(inv.sum())} tokens; median {inv.median():.0f}/page)")
    print(f"  wrapper collapse   : {100 * wrp.sum() / tot:5.1f}%  "
          f"({int(wrp.sum())} tokens; median {wrp.median():.0f}/page)")
    print("  ^ the split the census could NOT predict: invisible-subtree text "
          "is hidden from G3, so it is removable and was counted as untouchable")
    if "n_candidate_edits" in ok:
        cand = pd.to_numeric(ok["n_candidate_edits"], errors="coerce")
        kept = pd.to_numeric(ok["n_kept"], errors="coerce")
        drop = pd.to_numeric(ok.get("n_prescreen_dropped"), errors="coerce").fillna(0)
        print(f"\nedits kept: {int(kept.sum())}/{int(cand.sum())} "
              f"({100 * kept.sum() / max(cand.sum(), 1):.1f}%)  "
              f"-- {int(drop.sum())} dropped by the in-page prescreen (no "
              f"render), the rest blacklisted by bisection")
        if "gate_calls" in ok:
            capped = (pd.to_numeric(ok["gate_calls"], errors="coerce") >= 40).sum()
            print(f"pages at the 40-call ceiling: {int(capped)}")
    if "stamp_roundtrip_identical" in d:
        s = d["stamp_roundtrip_identical"].dropna()
        if len(s):
            print(f"stamped round-trip pixel-identical to original: "
                  f"{int(s.astype(bool).sum())}/{len(s)}  (was never checked before)")
        print(f"gate calls per page: median "
              f"{pd.to_numeric(ok['gate_calls']).median():.0f} "
              f"(whole-page-only would have been 1, and would have lost "
              f"every partially-bad page)")
    print(f"\n[06] wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--budget", type=int, default=40,
                    help="max gate calls (renders) per page during bisection")
    ap.add_argument("--on-level1", action="store_true", default=True,
                    help="compose on Level 1 output when present (default)")
    ap.add_argument("--on-original", dest="on_level1", action="store_false",
                    help="run Level 2 on the original HTML instead")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    t0 = time.time()
    with RenderHarness() as h:
        out = run_source(args.source, args, h, cfg, tk)
        if out:
            summarize(out, args.source)
    print(f"[06] total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()