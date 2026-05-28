#!/usr/bin/env bash
# Bootstrap the sibling repos under external/ that the v2 pipeline still needs.
#
# What v2 NEEDS:
#   * external/AIC24_Track1_YACHIYO_RIIPS  → SCT/MCT (untouched upstream tracker)
#   * external/TrackEval                   → HOTA/IDF1/MOTA scoring (pip package
#                                            lacks the runner scripts we use)
#
# What v2 NO LONGER NEEDS (removed from this script):
#   * BoT-SORT          — replaced by aic24_nvidia/models/detect_yolo.py (YOLO11)
#   * deep-person-reid  — replaced by aic24_nvidia/models/reid_solider.py (SOLIDER)
#   * mmpose / .venv-pose — replaced by aic24_nvidia/models/pose_rtmpose.py (rtmlib)
#
# Idempotent: each clone is skipped if already present.

set -euo pipefail

cd "$(dirname "$0")/.."
EXTERNAL="$(pwd)/external"
mkdir -p "$EXTERNAL"

# Upstream tracker — still required for SCT/MCT.
test -d "$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS" \
    || git clone https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS.git \
       "$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS"

# TrackEval — required by the evaluate stage. We use it via the cloned source
# (the pip package lacks the runner scripts we depend on). Patched
# np.float/int/bool → built-ins after clone (see CLAUDE.md "Upstream patches").
test -d "$EXTERNAL/TrackEval" \
    || git clone https://github.com/JonathonLuiten/TrackEval.git "$EXTERNAL/TrackEval"

echo "Bootstrap done."
echo "Detect/reid/pose models are pip-installed / vendored — no sibling clones needed."
echo "If the upstream patches in CLAUDE.md ('Upstream patches applied') are not in"
echo "place, re-apply them (only YACHIYO and TrackEval items 2-4 remain relevant)."
