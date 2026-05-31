#!/usr/bin/env bash
# Archive a finished pipeline run to OneDrive (or any rclone remote) and free the
# local copy — for when outputs/ is eating your disk.
#
# WHY THIS SHAPE:
#   * A run dir is ~3-4 GB across *tens of thousands* of small files (frames,
#     per-frame detections/embeddings). OneDrive throttles small-file storms
#     (HTTP 429) and a live FUSE mount of it is slow/fragile — so we pack each
#     run into ONE stream and push that instead of thousands of objects.
#   * The disk is usually nearly full (that's *why* you're archiving), so we
#     NEVER stage a multi-GB tarball on local disk: we stream
#         tar -> zstd -> rclone rcat
#     straight to the remote, then verify by streaming it back through `zstd -t`
#     (no local scratch space either), and only then delete the local run.
#   * After a run is archived its local dir is replaced by a small
#     `<run_id>.archived.json` stub recording where it went — `restore_run.sh`
#     reads that to pull it back.
#
# Usage:
#   scripts/archive_run.sh <run_id> [<run_id> ...]
#   scripts/archive_run.sh v2_solider scene041 --yes
#
# Flags:
#   --yes           delete the local run without the interactive confirm
#   --keep-local    upload + verify but DON'T delete the local copy (backup mode)
#   --no-verify     skip the stream-back integrity check (faster, less safe)
#   --force         allow archiving "baseline" (NORMALLY REFUSED — experiments/
#                   symlinks its upstream stages from outputs/baseline/)
#   --remote NAME   rclone remote   (default: $AIC24_RCLONE_REMOTE or "onedrive")
#   --dest PATH     remote dir      (default: aic24/outputs-archive)
#
# Restore later with: scripts/restore_run.sh <run_id>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS="$REPO_ROOT/outputs"
PY="python"; [ -x "$REPO_ROOT/.venv/bin/python" ] && PY="$REPO_ROOT/.venv/bin/python"
REMOTE="${AIC24_RCLONE_REMOTE:-onedrive}"
DEST="aic24/outputs-archive"
ASSUME_YES=0
KEEP_LOCAL=0
VERIFY=1
FORCE=0
RUN_IDS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --yes)        ASSUME_YES=1 ;;
    --keep-local) KEEP_LOCAL=1 ;;
    --no-verify)  VERIFY=0 ;;
    --force)      FORCE=1 ;;
    --remote)     REMOTE="$2"; shift ;;
    --dest)       DEST="$2"; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    -*)           echo "unknown flag: $1" >&2; exit 2 ;;
    *)            RUN_IDS+=("$1") ;;
  esac
  shift
done

[ ${#RUN_IDS[@]} -gt 0 ] || { echo "usage: $0 <run_id> [<run_id> ...] [flags]" >&2; exit 2; }
command -v rclone >/dev/null || { echo "rclone not installed" >&2; exit 1; }
command -v zstd   >/dev/null || { echo "zstd not installed"   >&2; exit 1; }

# Pre-flight: the remote must exist. If not, tell the user how to make it.
if ! rclone listremotes | grep -qx "${REMOTE}:"; then
  cat >&2 <<EOF
No rclone remote named "${REMOTE}:" is configured. Set one up once with:

    rclone config
    # n) New remote   →  name: ${REMOTE}
    # storage: onedrive   (Microsoft OneDrive)
    # client_id / client_secret: leave blank
    # region: global (1)   →  it opens a browser to sign in
    # then pick your drive (OneDrive Personal / Business) and confirm

Then re-run this script. (Override the name with --remote NAME.)
EOF
  exit 3
fi

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

for run_id in "${RUN_IDS[@]}"; do
  echo "==> ${run_id}"
  dir="$OUTPUTS/$run_id"
  stub="$OUTPUTS/$run_id.archived.json"
  name="$run_id.tar.zst"
  remote_path="$REMOTE:$DEST/$name"

  if [ "$run_id" = "baseline" ] && [ "$FORCE" -ne 1 ]; then
    echo "   REFUSED: outputs/baseline/ is the locked baseline that experiments/ symlinks" >&2
    echo "   its upstream stages from. Moving it breaks the experiment harness. (--force to override.)" >&2
    continue
  fi
  if [ ! -d "$dir" ]; then
    if [ -f "$stub" ]; then echo "   already archived (stub present) — skipping"; continue; fi
    echo "   no such run: $dir" >&2; continue
  fi

  nfiles=$(find "$dir" -type f | wc -l)
  nbytes=$(du -sb "$dir" | cut -f1)
  echo "   $(human "$nbytes") across ${nfiles} files  →  ${remote_path}"

  # Stream straight to the remote — NO local staging file (disk is tight).
  echo "   uploading (tar | zstd -T0 -3 | rclone rcat)..."
  tar -C "$OUTPUTS" -cf - "$run_id" | zstd -q -T0 -3 | rclone rcat "$remote_path" -P

  remote_bytes=$(rclone size --json "$remote_path" | python -c 'import sys,json;print(json.load(sys.stdin)["bytes"])')
  echo "   uploaded: $(human "$remote_bytes")"
  [ "$remote_bytes" -gt 0 ] || { echo "   ERROR: remote object is empty — NOT deleting local" >&2; exit 1; }

  if [ "$VERIFY" -eq 1 ]; then
    echo "   verifying (stream back through zstd -t)..."
    rclone cat "$remote_path" | zstd -t -  # exits non-zero (set -e) if corrupt/truncated
    echo "   integrity OK"
  fi

  # Record where it went BEFORE deleting, so an interrupted delete still leaves a pointer.
  archived_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "$stub" <<EOF
{
  "run_id": "$run_id",
  "remote": "$REMOTE",
  "dest": "$DEST",
  "name": "$name",
  "remote_path": "$remote_path",
  "source_bytes": $nbytes,
  "source_files": $nfiles,
  "archive_bytes": $remote_bytes,
  "verified": $( [ "$VERIFY" -eq 1 ] && echo true || echo false ),
  "archived_at": "$archived_at"
}
EOF

  # Capture the results in the durable ledger BEFORE the bytes leave local disk,
  # so an archived run is never a hole in results/README.md.
  "$PY" "$REPO_ROOT/scripts/results.py" add "$run_id" \
      --archived-remote "$REMOTE" --archived-path "$remote_path" \
      || echo "   (results-ledger update skipped — run 'python scripts/results.py scan' later)"

  if [ "$KEEP_LOCAL" -eq 1 ]; then
    echo "   --keep-local: local copy kept at $dir (stub written: $stub)"
    continue
  fi

  do_delete=$ASSUME_YES
  if [ "$do_delete" -ne 1 ]; then
    if [ -t 0 ]; then
      read -r -p "   delete local $dir to reclaim $(human "$nbytes")? [y/N] " ans
      [[ "$ans" =~ ^[Yy]$ ]] && do_delete=1
    else
      echo "   not a TTY and no --yes: keeping local copy. Re-run with --yes to reclaim space."
    fi
  fi

  if [ "$do_delete" -eq 1 ]; then
    rm -rf "$dir"
    echo "   freed $(human "$nbytes") — local dir removed, stub at $stub"
  else
    rm -f "$stub"  # didn't delete → don't leave a misleading "archived" stub
    echo "   kept local; remote copy is at ${remote_path} (no stub written)"
  fi
done

echo "done."
df -h "$OUTPUTS" | tail -1
