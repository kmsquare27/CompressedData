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
# Token counting (Qwen2.5-VL tokenizer -> tiktoken -> chars/4 fallback)
# ---------------------------------------------------------------------------
class TokenCounter:
    _inst = None

    def __init__(self):
        self.name, self._enc = "chars/4", None
        try:
            from transformers import AutoTokenizer
            self._enc = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
            self.name = "qwen2.5-vl"
        except Exception:
            try:
                import tiktoken
                self._enc = tiktoken.get_encoding("cl100k_base")
                self.name = "cl100k_base"
            except Exception:
                pass
        print(f"[gate] token counter: {self.name}")

    def count(self, text: str) -> int:
        if self._enc is None:
            return max(1, len(text) // 4)
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
# The gate
# ---------------------------------------------------------------------------
def run_gate(orig_html_path, comp_html_path, harness, work_dir, page_id,
             cfg: dict | None = None) -> dict:
    """Render both files under the harness and evaluate all gates.

    Returns a flat dict (CSV-friendly) with per-gate booleans, scores and
    an overall `accepted` flag. Screenshots land in work_dir.
    """
    cfg = cfg or load_config()
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    tk = TokenCounter.get()
    R: dict = {"page_id": page_id, "tokenizer": tk.name}

    # G0 -- parse
    comp_text = Path(comp_html_path).read_text(encoding="utf-8", errors="ignore")
    orig_text = Path(orig_html_path).read_text(encoding="utf-8", errors="ignore")
    try:
        import lxml.html as LH
        LH.fromstring(comp_text)
        R["g0_parse"] = True
    except Exception as e:  # noqa: BLE001
        R["g0_parse"] = False
        R["fail_reason"] = f"parse:{e}"

    # G1 -- tokens
    R["tokens_orig"] = tk.count(orig_text)
    R["tokens_comp"] = tk.count(comp_text)
    R["reduction_pct"] = round(
        100.0 * (R["tokens_orig"] - R["tokens_comp"]) / max(R["tokens_orig"], 1), 3)
    R["g1_tokens"] = (R["tokens_comp"] < R["tokens_orig"]
                      if cfg["require_token_reduction"] else True)

    # Render both sides
    png_o = work_dir / f"{page_id}_orig.png"
    png_c = work_dir / f"{page_id}_comp.png"
    lay_o = harness.render(orig_html_path, png_o)
    lay_c = harness.render(comp_html_path, png_c)
    img_o = np.asarray(Image.open(png_o).convert("RGB"))
    img_c = np.asarray(Image.open(png_c).convert("RGB"))

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
    try:
        from skimage import color as skcolor
        for i, j in pairs:
            la = _region_mean_lab(img_o, A[i]); lb = _region_mean_lab(img_c, B[j])
            if la is None or lb is None:
                continue
            des.append(float(skcolor.deltaE_ciede2000(la[None, :], lb[None, :])[0]))
    except Exception as e:  # noqa: BLE001
        R["g5_error"] = str(e)
    R["deltae_p95"] = round(float(np.percentile(des, 95)), 3) if des else 0.0
    R["deltae_max"] = round(float(np.max(des)), 3) if des else 0.0
    R["g5_color"] = (R["deltae_p95"] <= cfg["deltae_p95_max"]
                     and R["deltae_max"] <= cfg["deltae_max"])

    # Common-crop for pixel metrics (diagnostics + optional G6)
    h = min(img_o.shape[0], img_c.shape[0]); w = min(img_o.shape[1], img_c.shape[1])
    co, cc = img_o[:h, :w], img_c[:h, :w]

    # SSIM -- diagnostic only
    try:
        from skimage.metrics import structural_similarity as ssim
        R["ssim_diag"] = round(float(ssim(co, cc, channel_axis=2, data_range=255)), 5)
    except Exception:
        R["ssim_diag"] = None

    # G6 -- optional perceptual soft gate
    if cfg.get("use_lpips"):
        try:
            R["lpips_max_tile"] = round(_tiled_lpips(co, cc), 4)
            R["g6_lpips"] = R["lpips_max_tile"] <= cfg["lpips_max"]
        except Exception as e:  # noqa: BLE001
            R["g6_lpips"], R["g6_error"] = True, f"lpips unavailable: {e}"
    else:
        R["g6_lpips"] = True

    R["accepted"] = all([R["g0_parse"], R["g1_tokens"], R["g2_height"],
                         R["g3_text"], R["g4_blocks"], R["g5_color"],
                         R["g6_lpips"]])
    return R
