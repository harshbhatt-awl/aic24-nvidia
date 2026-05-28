"""Behavioural test: the SOLIDER ReID extractor must produce *discriminative*
embeddings — crops of the SAME person must be more similar than crops of
DIFFERENT people.

This guards against the feature-collapse bug where the extractor emitted a
near-constant vector (random different crops had cosine ~0.99), which destroyed
downstream YACHIYO association (HOTA/IDF1 collapse) while leaving per-frame
detection metrics intact.

Skips automatically when the real weights / frame data are not present, so it is
a no-op in data-less CI but runs as a real gate when validating the model.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
_WEIGHTS = _REPO / "weights" / "solider_swin_small.pth"
_RUN = _REPO / "outputs" / "merged_full" / "adapted" / "Original" / "scene_001" / "camera_0390"
_GT = _RUN / "gt" / "gt.txt"
_FRAMES = _RUN / "Frame"

pytestmark = pytest.mark.skipif(
    not (_WEIGHTS.exists() and _GT.exists() and _FRAMES.exists()),
    reason="SOLIDER weights or real frame/GT data not available",
)


def _load_id_crops(n_ids: int = 6, per_id: int = 3, frame_gap: int = 60):
    """Return {gt_id: [PIL crops]} sampling frames spread >= frame_gap apart."""
    rows_by_id: dict[int, list] = defaultdict(list)
    for line in _GT.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        fr, pid, x, y, w, h = (int(float(v)) for v in p[:6])
        rows_by_id[pid].append((fr, x, y, w, h))

    crops: dict[int, list] = {}
    for pid, rows in sorted(rows_by_id.items()):
        rows.sort()
        picked, last_fr = [], -(10**9)
        for fr, x, y, w, h in rows:
            if fr - last_fr < frame_gap:
                continue
            img_path = _FRAMES / f"{fr:06d}.jpg"
            if not img_path.exists() or w < 10 or h < 20:
                continue
            crop = Image.open(img_path).convert("RGB").crop((x, y, x + w, y + h))
            picked.append(crop)
            last_fr = fr
            if len(picked) == per_id:
                break
        if len(picked) == per_id:
            crops[pid] = picked
        if len(crops) == n_ids:
            break
    return crops


def test_solider_embeddings_are_discriminative():
    from aic24_nvidia.models import reid_solider

    crops = _load_id_crops()
    assert len(crops) >= 4, f"need >=4 identities with enough crops, got {len(crops)}"

    embs: dict[int, list[np.ndarray]] = {}
    for pid, imgs in crops.items():
        vs = []
        for im in imgs:
            v = reid_solider._embed(im).astype(np.float64)
            vs.append(v / (np.linalg.norm(v) + 1e-12))
        embs[pid] = vs

    same, diff = [], []
    for pid, vs in embs.items():
        for a, b in itertools.combinations(vs, 2):
            same.append(float(a @ b))
    for p1, p2 in itertools.combinations(embs, 2):
        for a in embs[p1]:
            for b in embs[p2]:
                diff.append(float(a @ b))

    same_m, diff_m = float(np.mean(same)), float(np.mean(diff))
    print(f"\nsame-id cosine mean={same_m:.3f}  different-id cosine mean={diff_m:.3f}  "
          f"margin={same_m - diff_m:.3f}")

    # Not degenerate: different people must not all look identical.
    assert diff_m < 0.90, f"embeddings collapsed: different-id cosine {diff_m:.3f} >= 0.90"
    # Discriminative: same person clearly more similar than different people.
    assert same_m - diff_m > 0.10, (
        f"insufficient separation: same={same_m:.3f} diff={diff_m:.3f} "
        f"margin={same_m - diff_m:.3f} <= 0.10"
    )
