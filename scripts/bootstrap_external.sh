#!/usr/bin/env bash
# Bootstrap external sibling repos required by AIC24_Track1_YACHIYO_RIIPS.
# Idempotent: skips clones already present.
set -euo pipefail

cd "$(dirname "$0")/.."
EXTERNAL="$(pwd)/external"
mkdir -p "$EXTERNAL"

# Upstream already cloned by Task 11; double-check.
test -d "$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS" \
    || git clone https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS.git \
       "$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS"

# BoT-SORT (detection runtime)
test -d "$EXTERNAL/BoT-SORT" \
    || git clone https://github.com/NirAharon/BoT-SORT.git "$EXTERNAL/BoT-SORT"

# deep-person-reid (embedding runtime)
test -d "$EXTERNAL/deep-person-reid" \
    || git clone https://github.com/KaiyangZhou/deep-person-reid.git \
       "$EXTERNAL/deep-person-reid"

# mmpose 0.x (pose runtime — pinned to last 0.x release)
test -d "$EXTERNAL/mmpose" \
    || ( git clone https://github.com/open-mmlab/mmpose.git "$EXTERNAL/mmpose" \
         && git -C "$EXTERNAL/mmpose" checkout v0.29.0 )

# Copy the upstream's custom Python files into the sibling repos.
UPSTREAM="$EXTERNAL/AIC24_Track1_YACHIYO_RIIPS"
cp "$UPSTREAM/detector/aic24_get_detection.py" "$EXTERNAL/BoT-SORT/tools/"
cp "$UPSTREAM/embedder/aic24_extract.py"        "$EXTERNAL/deep-person-reid/torchreid/"
cp "$UPSTREAM/poser/load_tracking_result.py"     "$EXTERNAL/mmpose/demo/"
cp "$UPSTREAM/poser/top_down_video_demo_with_track_file.py" "$EXTERNAL/mmpose/demo/"

echo "Bootstrap done."
echo "Next: install the three sibling repos' Python deps (see their READMEs)."
echo "Then place YOLOX checkpoint at $EXTERNAL/BoT-SORT/bytetrack_x_mot17.pth.tar"
