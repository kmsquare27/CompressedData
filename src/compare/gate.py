"""Acceptance Gate v2.

A compressed page is accepted only if ALL hard gates pass:
  G0  compressed HTML parses
  G1  token count strictly reduced (target-model tokenizer)
  G2  full-page render height within tolerance (NEVER resize screenshots)
  G3  visible text identical (normalized Levenshtein ratio)
  G4  visual blocks match 1:1 with high IoU and small center shift
  G5  per-matched-block color difference (CIEDE2000) within tolerance
  G6  (soft, optional) tiled LPIPS perceptual score
SSIM is computed as a REPORTED DIAGNOSTIC only -- it does not gate.

Thresholds live in config/gate_config.yaml and are PROVISIONAL until the
mutation stress test (03_run_stress_test.py) calibrates them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "height_tol_px": 2,
    "text_ratio_min": 0.995,
    "iou_mean_min": 0.95,
    "center_shift_max_px": 5,
    "deltae_p95_max": 2.0,
    "deltae_max": 5.0,
    "require_token_reduction": True,
    "use_lpips": False,
    "lpips_max": 0.03,
}


def load_config(path="config/gate_config.yaml") -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(path)
    if p.exists():
        try:
            import yaml
            cfg.update(yaml.safe_load(p.read_text()) or {})
        except Exception as e:  # noqa: BLE001
            print(f"[gate] could not read {path} ({e}); using defaults")
    return cfg


# ---------------------------------------------------------------------------
# Token counting (Qwen2.5-VL tokenizer -- the target model's own tokenizer,
# no fallback: a substitute tokenizer's counts are not the number that
# matters, and reporting them as if they were would be silently wrong)
# ---------------------------------------------------------------------------
class TokenCounter:
    _inst = None
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self):
        from transformers import AutoTokenizer
        try:
            self._enc = AutoTokenizer.from_pretrained(
                self.MODEL_ID, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"[gate] could not load the tokenizer for {self.MODEL_ID}; token "
                f"counts have no fallback (fix network access / the local HF "
                f"cache and retry): {e}") from e
        revision = "unknown"
        try:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(self.MODEL_ID).sha
        except Exception:  # noqa: BLE001
            pass
        self.name = f"qwen2.5-vl@{revision}"
        print(f"[gate] token counter: {self.name}")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    @classmethod
    def get(cls) -> "TokenCounter":
        if cls._inst is None:
            cls._inst = TokenCounter()
        return cls._inst


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _norm_text(t: str) -> str:
    return " ".join((t or "").split())


def _text_ratio(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except Exception:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


def _iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _center_shift(a: dict, b: dict) -> float:
    return max(abs((a["x"] + a["w"] / 2) - (b["x"] + b["w"] / 2)),
               abs((a["y"] + a["h"] / 2) - (b["y"] + b["h"] / 2)))


def _match_boxes(A: list, B: list):
    """Match boxes across renders. Same-text pairs are strongly preferred.
    Returns (pairs, n_unmatched)."""
    if not A and not B:
        return [], 0
    if not A or not B:
        return [], abs(len(A) - len(B))
    cost = np.zeros((len(A), len(B)), dtype=float)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            c = 1.0 - _iou(a, b)
            if _norm_text(a["text"]) != _norm_text(b["text"]):
                c += 1.0
            cost[i, j] = c
    try:
        from scipy.optimize import linear_sum_assignment
        ri, cj = linear_sum_assignment(cost)
        pairs = list(zip(ri.tolist(), cj.tolist()))
    except Exception:  # greedy fallback
        pairs, used = [], set()
        for i in range(len(A)):
            j = int(np.argmin([cost[i, j] if j not in used else 9e9
                               for j in range(len(B))]))
            pairs.append((i, j)); used.add(j)
    n_unmatched = abs(len(A) - len(B))
    return pairs, n_unmatched


def _region_mean_lab(img: np.ndarray, box: dict):
    from skimage import color as skcolor
    H, W = img.shape[:2]
    x1 = max(0, int(round(box["x"])));  y1 = max(0, int(round(box["y"])))
    x2 = min(W, int(round(box["x"] + box["w"])))
    y2 = min(H, int(round(box["y"] + box["h"])))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    crop = img[y1:y2, x1:x2, :3].astype(np.float64) / 255.0
    return skcolor.rgb2lab(crop).reshape(-1, 3).mean(axis=0)


def _ssim_rgb_lowmem(a: np.ndarray, b: np.ndarray, strip_rows: int = 512):
    """Mean-over-channels SSIM in bounded memory, bit-exact vs the naive call.

    scikit-image promotes uint8 to float64 and holds ~16 full-size temporaries
    while filtering, so peak memory grows with page height: ~465 MB for a
    1280x2836 render. The stress test calls SSIM twice per sample, which is
    what exhausted a real 100-page WebCode2M run at page 87.

    SSIM at a pixel depends only on its 7x7 filter neighbourhood, so a
    horizontal strip extended by the filter radius yields *identical* values
    for that strip's interior rows. Summing interior contributions strip by
    strip therefore reproduces skimage's border-cropped mean exactly (verified
    |delta| = 0.0 to 7 decimals) while capping peak memory at strip size --
    ~42 MB here, independent of how tall the page is.

    Degrades to luma, then to None: SSIM is a diagnostic column, never a gate,
    so it must never be able to abort a multi-hour run.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None

    def plane(a1, b1, win=7):
        pad = (win - 1) // 2
        H = a1.shape[0]
        if H <= 2 * pad + 1:
            return float(ssim(np.ascontiguousarray(a1, dtype=np.float32),
                              np.ascontiguousarray(b1, dtype=np.float32),
                              data_range=255.0))
        tot, n = 0.0, 0
        for r0 in range(0, H, strip_rows):
            r1 = min(H, r0 + strip_rows)
            e0, e1 = max(0, r0 - pad), min(H, r1 + pad)
            _, S = ssim(np.ascontiguousarray(a1[e0:e1], dtype=np.float32),
                        np.ascontiguousarray(b1[e0:e1], dtype=np.float32),
                        data_range=255.0, full=True)
            lo, hi = max(r0, pad) - e0, min(r1, H - pad) - e0
            if hi > lo:
                sub = S[lo:hi, pad:S.shape[1] - pad]
                tot += float(sub.sum()); n += sub.size
                del sub
            del S
        return tot / n if n else float("nan")

    try:
        return round(sum(plane(a[:, :, i], b[:, :, i]) for i in range(3)) / 3.0, 5)
    except MemoryError:
        pass
    except Exception:  # e.g. degenerate extent smaller than the filter window
        return None
    try:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return round(plane(a[:, :, :3].astype(np.float32) @ w,
                           b[:, :, :3].astype(np.float32) @ w), 5)
    except Exception:
        return None


def _tiled_lpips(a: np.ndarray, b: np.ndarray, tile_h: int = 800) -> float:
    import lpips, torch
    net = getattr(_tiled_lpips, "_net", None)
    if net is None:
        net = lpips.LPIPS(net="alex", verbose=False)
        _tiled_lpips._net = net

    def to_t(x):
        t = torch.from_numpy(x[:, :, :3].copy()).permute(2, 0, 1).float()
        return (t / 127.5 - 1.0).unsqueeze(0)

    worst = 0.0
    for y0 in range(0, a.shape[0], tile_h):
        ta, tb = a[y0:y0 + tile_h], b[y0:y0 + tile_h]
        if ta.shape[0] < 32:
            continue
        with torch.no_grad():
            worst = max(worst, float(net(to_t(ta), to_t(tb)).item()))
    return worst


# ---------------------------------------------------------------------------
# The gate, decomposed: render_page() + evaluate_pair()
# ---------------------------------------------------------------------------
# `run_gate` used to render and compare in one call. The two halves are now
# separate because the mutation stress test (a) must exercise EXACTLY the
# comparison code that gates production pages -- a stress test that
# re-implements the metrics validates a copy, not the gate -- and (b) renders
# the original once per page and evaluates ~25 mutants against that cached
# render. `run_gate` keeps its exact signature, behavior and output keys.

class PageArtifacts:
    """One rendered page: everything evaluate_pair needs to compare it."""
    __slots__ = ("html_path", "png_path", "html_text", "tokens", "layout", "img")

    def __init__(self, html_path, png_path, html_text, tokens, layout, img):
        self.html_path = str(html_path)
        self.png_path = str(png_path)
        self.html_text = html_text
        self.tokens = tokens
        self.layout = layout      # {boxes, docW, docH, text} from the harness
        self.img = img            # np.uint8 RGB array of the full-page render


def render_page(harness, html_path, png_path, token_counter=None) -> PageArtifacts:
    """Render one HTML file under the harness and bundle all gate inputs."""
    tk = token_counter or TokenCounter.get()
    html_text = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    layout = harness.render(html_path, png_path)
    img = np.asarray(Image.open(png_path).convert("RGB"))
    return PageArtifacts(html_path, png_path, html_text, tk.count(html_text),
                         layout, img)


def evaluate_pair(orig: PageArtifacts, comp: PageArtifacts,
                  cfg: dict | None = None, page_id: str = "") -> dict:
    """Evaluate every gate on two already-rendered pages. Pure comparison:
    no rendering happens here. Same output dict as run_gate."""
    cfg = cfg or load_config()
    R: dict = {"page_id": page_id, "tokenizer": TokenCounter.get().name}

    # G0 -- parse
    try:
        import lxml.html as LH
        LH.fromstring(comp.html_text)
        R["g0_parse"] = True
    except Exception as e:  # noqa: BLE001
        R["g0_parse"] = False
        R["fail_reason"] = f"parse:{e}"

    # G1 -- tokens
    R["tokens_orig"] = orig.tokens
    R["tokens_comp"] = comp.tokens
    R["reduction_pct"] = round(
        100.0 * (R["tokens_orig"] - R["tokens_comp"]) / max(R["tokens_orig"], 1), 3)
    R["g1_tokens"] = (R["tokens_comp"] < R["tokens_orig"]
                      if cfg["require_token_reduction"] else True)

    lay_o, lay_c = orig.layout, comp.layout
    img_o, img_c = orig.img, comp.img

    # G2 -- geometry (never resize; mismatch is evidence)
    R["docH_orig"], R["docH_comp"] = lay_o["docH"], lay_c["docH"]
    R["height_delta_px"] = abs(lay_o["docH"] - lay_c["docH"])
    R["g2_height"] = (R["height_delta_px"] <= cfg["height_tol_px"]
                      and img_o.shape[1] == img_c.shape[1])

    # G3 -- text integrity
    ta, tb = _norm_text(lay_o["text"]), _norm_text(lay_c["text"])
    R["text_ratio"] = round(_text_ratio(ta, tb), 5)
    R["g3_text"] = R["text_ratio"] >= cfg["text_ratio_min"]

    # G4 -- block match + IoU + center shift
    A, B = lay_o["boxes"], lay_c["boxes"]
    R["n_boxes_orig"], R["n_boxes_comp"] = len(A), len(B)
    pairs, n_unmatched = _match_boxes(A, B)
    ious = [_iou(A[i], B[j]) for i, j in pairs]
    shifts = [_center_shift(A[i], B[j]) for i, j in pairs]
    R["iou_mean"] = round(float(np.mean(ious)), 4) if ious else 1.0
    R["center_shift_max"] = round(float(np.max(shifts)), 2) if shifts else 0.0
    R["n_unmatched"] = n_unmatched
    R["g4_blocks"] = (n_unmatched == 0
                      and R["iou_mean"] >= cfg["iou_mean_min"]
                      and R["center_shift_max"] <= cfg["center_shift_max_px"])

    # G5 -- color (CIEDE2000 on matched block regions)
    des = []
    g5_exc = None
    try:
        from skimage import color as skcolor
        for i, j in pairs:
            la = _region_mean_lab(img_o, A[i]); lb = _region_mean_lab(img_c, B[j])
            if la is None or lb is None:
                continue
            des.append(float(skcolor.deltaE_ciede2000(la[None, :], lb[None, :])[0]))
    except Exception as e:  # noqa: BLE001
        g5_exc = e
    if g5_exc is not None or not des:
        # No measurement is not a pass: 0.0 would silently clear both
        # thresholds. Record why and let the soundness rule (pixel-identical
        # renders) be the only thing that can still accept this page.
        R["g5_color"] = False
        R["g5_error"] = str(g5_exc) if g5_exc is not None else "no color measurements"
    else:
        R["deltae_p95"] = round(float(np.percentile(des, 95)), 3)
        R["deltae_max"] = round(float(np.max(des)), 3)
        R["g5_color"] = (R["deltae_p95"] <= cfg["deltae_p95_max"]
                         and R["deltae_max"] <= cfg["deltae_max"])

    # Common-crop for pixel metrics (diagnostics + optional G6)
    h = min(img_o.shape[0], img_c.shape[0]); w = min(img_o.shape[1], img_c.shape[1])
    co, cc = img_o[:h, :w], img_c[:h, :w]

    # SSIM -- diagnostic only. Per-channel float32 (see _ssim_rgb_lowmem):
    # full-page renders of tall pages make the float64/3-channel path a
    # memory hazard, and a diagnostic column must never abort a run.
    try:
        R["ssim_diag"] = _ssim_rgb_lowmem(co, cc)
    except Exception:
        R["ssim_diag"] = None

    # G6 -- optional perceptual soft gate
    if cfg.get("use_lpips"):
        try:
            R["lpips_max_tile"] = round(_tiled_lpips(co, cc), 4)
            R["g6_lpips"] = R["lpips_max_tile"] <= cfg["lpips_max"]
        except Exception as e:  # noqa: BLE001
            R["g6_lpips"], R["g6_error"] = False, f"lpips unavailable: {e}"
    else:
        R["g6_lpips"] = True

    # ---- SOUNDNESS RULE ------------------------------------------------
    # Identical rasters mean identical appearance, full stop. Every gate
    # metric here is derived from the DOM (box rects, computed text), and no
    # DOM-derived quantity is a faithful proxy for visual change: an opaque
    # white <div> on a white page, or a transparent text block, can widen by
    # hundreds of pixels without altering a single rendered pixel. Measured
    # on WebCode2M, that artifact produced center shifts of 492px and
    # CIEDE2000 of 6.3 on renders that were byte-identical, and it was the
    # sole cause of every safe false-rejection in the 87-page stress run.
    # So the raster wins: when it is unchanged, the structural gates cannot
    # veto. Metrics are still recorded (unmodified) for transparency, and
    # this can never mask a real breakage -- a mutation that changes nothing
    # on screen has, by construction, broken nothing on screen.
    R["pixel_identical"] = _pixel_identical(img_o, img_c)
    if R["pixel_identical"]:
        R["g2_height"] = R["g3_text"] = True
        R["g4_blocks"] = R["g5_color"] = R["g6_lpips"] = True

    R["accepted"] = all([R["g0_parse"], R["g1_tokens"], R["g2_height"],
                         R["g3_text"], R["g4_blocks"], R["g5_color"],
                         R["g6_lpips"]])
    return R


def _pixel_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a[:, :, :3], b[:, :, :3]))


# Visual gates only (G2-G6): what the stress test uses to score detection.
# G0 always passes on mutants (they parse) and G1 measures token direction,
# not vision -- style-adding mutants INCREASE tokens, so leaving G1 in the
# stress verdict would let the gate "detect" them for a non-visual reason
# and fake a perfect score. See STRESS_TEST.md.
VISUAL_GATE_KEYS = ("g2_height", "g3_text", "g4_blocks", "g5_color", "g6_lpips")


def visual_accepted(R: dict) -> bool:
    """Acceptance by the visual gates alone (G2-G6), ignoring G0/G1."""
    return all(bool(R.get(k, False)) for k in VISUAL_GATE_KEYS)


def run_gate(orig_html_path, comp_html_path, harness, work_dir, page_id,
             cfg: dict | None = None) -> dict:
    """Render both files under the harness and evaluate all gates.

    Returns a flat dict (CSV-friendly) with per-gate booleans, scores and
    an overall `accepted` flag. Screenshots land in work_dir.
    """
    cfg = cfg or load_config()
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    tk = TokenCounter.get()
    art_o = render_page(harness, orig_html_path,
                        work_dir / f"{page_id}_orig.png", tk)
    art_c = render_page(harness, comp_html_path,
                        work_dir / f"{page_id}_comp.png", tk)
    return evaluate_pair(art_o, art_c, cfg, page_id)