#!/usr/bin/env bash
# Make a freshly-rented GPU box pipeline-ready in one command — clone is assumed
# done (this script lives in the repo), so you run it FROM the repo root:
#
#     git clone https://github.com/harshbhatt-awl/aic24-nvidia && cd aic24-nvidia
#     scripts/remote_setup.sh
#
# WHY THIS SHAPE / what a fresh box is missing vs your laptop:
#   * Python 3.14 + torch 2.12 **+cu130** is the verified stack, but the lock pins
#     `torch==2.12.0` BARE (no +cu130, no index URL) — a plain `pip install -r`
#     silently pulls the wrong wheel. So we install torch/torchvision FIRST from
#     the cu130 index, then the rest of the lock (torch is then already satisfied).
#   * `uv` gets us the exact Python 3.14 without fighting the box's system python;
#     we auto-install it if absent (tiny, official script).
#   * The cu130 runtime needs an NVIDIA driver >= 580. We preflight `nvidia-smi`
#     and warn loudly if the host driver is too old (torch.cuda would be False).
#   * `data/nvidia_mtmc_2024` and `weights/solider_swin_small.pth` are SYMLINKS into
#     a sibling repo / worktree on the laptop — they don't exist here and both dirs
#     are gitignored. We materialize them best-effort from your rclone remote
#     (default onedrive:, same one archive_run.sh uses) and print instructions if
#     the remote copy isn't there. detect (yolo11x.pt) + pose (RTMPose ONNX)
#     auto-download on first run, so only the SOLIDER reid weight needs fetching.
#   * `external/{YACHIYO,TrackEval}` are cloned + patched by bootstrap_external.sh.
#
# Idempotent: re-running skips what's already in place (venv, siblings, data).
#
# Flags:
#   --no-apt         only CHECK system deps, don't apt-get install them
#   --skip-data      don't try to fetch the dataset (set it up yourself)
#   --skip-weights   don't try to fetch the SOLIDER reid weight
#   --remote NAME    rclone remote for data/weights (default: $AIC24_RCLONE_REMOTE or "onedrive")
#   -h | --help      show this header
#
# Env overrides:
#   AIC24_TORCH_INDEX          PyTorch wheel index   (default https://download.pytorch.org/whl/cu130)
#   AIC24_DATA_REMOTE_PATH     rclone path to dataset (default <remote>:aic24/data/nvidia_mtmc_2024)
#   AIC24_WEIGHTS_REMOTE_PATH  rclone path to weights (default <remote>:aic24/weights)
#
# After this finishes you run the pipeline as usual (in tmux so an SSH drop doesn't
# kill a ~45-min run):  tmux new -s aic24 ';' source .venv/bin/activate ';' python pipeline.py all --config configs/baseline.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${AIC24_RCLONE_REMOTE:-onedrive}"
TORCH_INDEX="${AIC24_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
DATA_REMOTE_PATH="${AIC24_DATA_REMOTE_PATH:-$REMOTE:aic24/data/nvidia_mtmc_2024}"
WEIGHTS_REMOTE_PATH="${AIC24_WEIGHTS_REMOTE_PATH:-$REMOTE:aic24/weights}"
VENV="$REPO_ROOT/.venv"
PYBIN="$VENV/bin/python"
DO_APT=1; DO_DATA=1; DO_WEIGHTS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-apt)       DO_APT=0 ;;
    --skip-data)    DO_DATA=0 ;;
    --skip-weights) DO_WEIGHTS=0 ;;
    --remote)       REMOTE="$2"; shift ;;
    -h|--help)      sed -n '2,46p' "$0"; exit 0 ;;
    -*)             echo "unknown flag: $1" >&2; exit 2 ;;
    *)              echo "unexpected arg: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*" >&2; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
say "GPU / driver preflight"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
  drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${drv_major:-0}" -lt 580 ] 2>/dev/null; then
    warn "driver major $drv_major < 580 — torch 2.12+cu130 (CUDA 13) needs driver >= 580."
    warn "torch.cuda.is_available() will likely be False. Rent a host with a newer driver,"
    warn "or override AIC24_TORCH_INDEX to a CUDA version this driver supports."
  else
    ok "driver $drv_major supports CUDA 13 (cu130)"
  fi
else
  warn "nvidia-smi not found — no GPU visible. Detect/reid will fall back to CPU (very slow)."
fi

# ---------------------------------------------------------------------------
say "System dependencies"
# ffmpeg is required by the `frames` stage; the rest mirror the archive/restore
# tooling and long-run ergonomics. build-essential is for any wheels lacking a
# manylinux build on py3.14.
PKGS=(git curl ca-certificates ffmpeg zstd rclone tmux build-essential)
missing=()
for p in git curl ffmpeg zstd rclone tmux; do command -v "$p" >/dev/null || missing+=("$p"); done
if [ ${#missing[@]} -eq 0 ]; then
  ok "all present (git curl ffmpeg zstd rclone tmux)"
elif [ "$DO_APT" -eq 1 ] && command -v apt-get >/dev/null; then
  SUDO=""; if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
  warn "installing: ${missing[*]}"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${PKGS[@]}"
  ok "system deps installed"
else
  warn "missing: ${missing[*]} — install them (apt-get install ${PKGS[*]}) and re-run, or pass --no-apt to ignore."
fi

# ---------------------------------------------------------------------------
say "Python 3.14 venv + cu130 torch (via uv)"
if ! command -v uv >/dev/null; then
  warn "uv not found — installing it (official script)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null || { echo "uv install failed; install it manually: https://docs.astral.sh/uv/" >&2; exit 1; }
fi

if [ ! -x "$PYBIN" ]; then
  uv venv "$VENV" --python 3.14
  ok "created $VENV (Python 3.14)"
else
  ok "$VENV already exists"
fi

# torch/torchvision FIRST from the cu130 index so the bare `torch==2.12.0` in the
# lock resolves to the +cu130 build (and is then already satisfied).
say "Installing torch 2.12.0 + torchvision 0.27.0 from $TORCH_INDEX"
uv pip install -p "$PYBIN" torch==2.12.0 torchvision==0.27.0 --index-url "$TORCH_INDEX"

say "Installing the rest of requirements.lock"
# --extra-index-url keeps the torch index available for any nvidia-*-cu13 deps;
# torch is already satisfied so it is not swapped for a PyPI wheel.
uv pip install -p "$PYBIN" -r requirements.lock --extra-index-url "$TORCH_INDEX"

say "Installing aic24_nvidia (editable)"
uv pip install -p "$PYBIN" -e . --no-deps
ok "environment built"

# ---------------------------------------------------------------------------
say "External siblings (YACHIYO + TrackEval, pinned + patched)"
bash "$REPO_ROOT/scripts/bootstrap_external.sh"

# ---------------------------------------------------------------------------
# Helper: pull from rclone remote if it's configured and the source exists.
# Returns 0 only on a successful copy.
rclone_pull() {  # <src remote_path> <dst local_dir>
  local src="$1" dst="$2"
  command -v rclone >/dev/null || { warn "rclone not installed — cannot pull $src"; return 1; }
  rclone listremotes 2>/dev/null | grep -qx "${REMOTE}:" || { warn "rclone remote '${REMOTE}:' not configured (run 'rclone config')"; return 1; }
  rclone lsf "$src" >/dev/null 2>&1 || { warn "not found on remote: $src"; return 1; }
  mkdir -p "$dst"
  rclone copy "$src" "$dst" -P
}

say "Dataset (data/nvidia_mtmc_2024)"
DATA_DIR="$REPO_ROOT/data/nvidia_mtmc_2024"
if [ -e "$DATA_DIR/MTMC_Tracking_2024" ]; then
  ok "dataset present"
elif [ "$DO_DATA" -eq 0 ]; then
  warn "--skip-data: set up data/nvidia_mtmc_2024 yourself (needs MTMC_Tracking_2024/ + Warehouse_001/)"
else
  if [ -L "$DATA_DIR" ]; then warn "removing dangling symlink $DATA_DIR"; rm -f "$DATA_DIR"; fi
  if rclone_pull "$DATA_REMOTE_PATH" "$DATA_DIR"; then
    ok "dataset pulled from $DATA_REMOTE_PATH"
  else
    warn "Could not auto-fetch the dataset. Either:"
    warn "  • stage it once:  rclone copy <laptop-or-onedrive> $DATA_REMOTE_PATH   then re-run, or"
    warn "  • download NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2024 (HuggingFace) into $DATA_DIR"
    warn "  Layout needed: $DATA_DIR/{MTMC_Tracking_2024/val/scene_044, Warehouse_001/{videos,calibration.json,ground_truth.json}}"
  fi
fi

say "Model weights (weights/solider_swin_small.pth)"
W_DIR="$REPO_ROOT/weights"
W_BASE="$W_DIR/solider_swin_small.pth"
mkdir -p "$W_DIR"
if [ -L "$W_BASE" ] && [ ! -e "$W_BASE" ]; then warn "removing dangling symlink $W_BASE"; rm -f "$W_BASE"; fi
if [ -r "$W_BASE" ]; then
  ok "SOLIDER reid weight present"
elif [ "$DO_WEIGHTS" -eq 0 ]; then
  warn "--skip-weights: put the SOLIDER Swin-Small weight at $W_BASE (reid stage needs it)"
else
  if rclone_pull "$WEIGHTS_REMOTE_PATH" "$W_DIR"; then
    [ -r "$W_BASE" ] && ok "weights pulled from $WEIGHTS_REMOTE_PATH" \
      || warn "pulled $WEIGHTS_REMOTE_PATH but $W_BASE still missing — check the remote layout"
  else
    warn "Could not auto-fetch weights. Get solider_swin_small.pth from the"
    warn "tinyvision/SOLIDER-REID release and place it at $W_BASE,"
    warn "or stage it once:  rclone copy <src> $WEIGHTS_REMOTE_PATH"
    warn "(detect + pose checkpoints auto-download on first run — only reid needs this.)"
  fi
fi

# ---------------------------------------------------------------------------
say "Sanity check"
"$PYBIN" - <<'PY' || warn "sanity check reported a problem (see above) — fix before a full run"
import torch
print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device: {torch.cuda.get_device_name(0)}")
else:
    print("    !! CUDA NOT available — check the driver (>=580 for cu130) / torch index")
import aic24_nvidia  # editable install importable?
print("    aic24_nvidia importable: OK")
PY

cat <<EOF

$(ok "remote box is set up")

Next steps on THIS box:
    source .venv/bin/activate
    tmux new -s aic24                 # so a dropped SSH session doesn't kill the run
    python pipeline.py all --config configs/baseline.yaml
    # detach with Ctrl-b d ; reattach later with: tmux attach -t aic24

From YOUR laptop (connect / view the dashboard):
    # ~/.ssh/config:  Host aic24-gpu / HostName <ip> / Port <port> / User root / IdentityFile ~/.ssh/id_ed25519
    # then in VS Code: "Remote-SSH: Connect to Host" → aic24-gpu (edits/runs live on the box)
    ssh -L 8501:localhost:8501 aic24-gpu     # then: python pipeline.py dashboard  → open http://localhost:8501

When done, reclaim/keep results before killing the box:
    scripts/archive_run.sh <run_id> --keep-local    # upload to ${REMOTE}: (results auto-recorded to the ledger)
EOF
