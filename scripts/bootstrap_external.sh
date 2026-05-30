#!/usr/bin/env bash
# Bootstrap the sibling repos under external/ that the pipeline needs, PINNED to
# known-good commits and with our patches applied automatically.
#
# What we NEED:
#   * external/AIC24_Track1_YACHIYO_RIIPS  → SCT/MCT (upstream tracker)
#   * external/TrackEval                   → HOTA/IDF1/MOTA scoring (pip package
#                                            lacks the runner scripts we use)
#
# What we NO LONGER NEED (not cloned):
#   * BoT-SORT          — replaced by aic24_nvidia/models/detect_yolo.py (YOLO11)
#   * deep-person-reid  — replaced by aic24_nvidia/models/reid_solider.py (SOLIDER)
#   * mmpose / .venv-pose — replaced by aic24_nvidia/models/pose_rtmpose.py (rtmlib)
#   (If stale clones of these still exist under external/ from an older setup,
#    they are unused and safe to `rm -rf`.)
#
# Patches are VENDORED under patches/ and applied on a fresh clone, so a re-clone
# no longer silently loses them (previously a documented manual step). Commit
# SHAs are pinned below so an upstream force-push or breaking change cannot creep
# in. After bootstrap, verify_patches.sh confirms the patches are in place.
#
# Idempotent: an existing clone is left untouched (clone + checkout + patch only
# run on a fresh clone).

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
EXTERNAL="$ROOT/external"
PATCHES="$ROOT/patches"
mkdir -p "$EXTERNAL"

# Pinned upstream commits (known-good with our patches).
YACHIYO_URL="https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS.git"
YACHIYO_SHA="f881fe0edf60fe697efb2bf4c935b80a80861230"
TRACKEVAL_URL="https://github.com/JonathonLuiten/TrackEval.git"
TRACKEVAL_SHA="12c8791b303e0a0b50f753af204249e622d0281a"

clone_pinned() {
    local url="$1" dst="$2" sha="$3" patch="$4"
    if [ -d "$dst" ]; then
        echo "already present: $dst (skipping clone + patch)"
        return 0
    fi
    echo "cloning $url -> $dst"
    git clone "$url" "$dst"
    echo "checking out pinned $sha"
    git -C "$dst" checkout --quiet "$sha"
    if [ -n "$patch" ] && [ -f "$patch" ]; then
        echo "applying patch: $patch"
        git -C "$dst" apply "$patch"
    fi
}

clone_pinned "$YACHIYO_URL"   "$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS" "$YACHIYO_SHA"   "$PATCHES/yachiyo.patch"
clone_pinned "$TRACKEVAL_URL" "$EXTERNAL/TrackEval"                  "$TRACKEVAL_SHA" "$PATCHES/trackeval.patch"

echo "Bootstrap done. Detect/reid/pose models are pip-installed / vendored."

# Confirm patches are present (fails loudly if a re-clone or upstream change
# dropped them).
bash "$ROOT/scripts/verify_patches.sh"
