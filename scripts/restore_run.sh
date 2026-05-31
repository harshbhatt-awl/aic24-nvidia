#!/usr/bin/env bash
# Restore a run previously archived with archive_run.sh back into outputs/.
#
# Streams the remote archive straight back in (rclone cat | zstd -d | tar -x) —
# no local scratch file needed, same as the archive side. Reads the
# `outputs/<run_id>.archived.json` stub (if present) to find the remote path.
#
# Usage:
#   scripts/restore_run.sh <run_id>
#   scripts/restore_run.sh v2_solider --purge-remote   # delete remote copy after restore
#
# Flags:
#   --force          overwrite an existing local outputs/<run_id>
#   --purge-remote   delete the remote archive once the local restore succeeds
#   --remote NAME    rclone remote   (default: stub's remote, else $AIC24_RCLONE_REMOTE or "onedrive")
#   --dest PATH      remote dir      (default: stub's dest, else aic24/outputs-archive)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS="$REPO_ROOT/outputs"
REMOTE="${AIC24_RCLONE_REMOTE:-onedrive}"
DEST="aic24/outputs-archive"
FORCE=0
PURGE=0
RUN_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force)        FORCE=1 ;;
    --purge-remote) PURGE=1 ;;
    --remote)       REMOTE="$2"; shift ;;
    --dest)         DEST="$2"; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    -*)             echo "unknown flag: $1" >&2; exit 2 ;;
    *)              RUN_ID="$1" ;;
  esac
  shift
done

[ -n "$RUN_ID" ] || { echo "usage: $0 <run_id> [flags]" >&2; exit 2; }
command -v rclone >/dev/null || { echo "rclone not installed" >&2; exit 1; }
command -v zstd   >/dev/null || { echo "zstd not installed"   >&2; exit 1; }

dir="$OUTPUTS/$RUN_ID"
stub="$OUTPUTS/$RUN_ID.archived.json"

# Prefer the stub's recorded location (the archive may have used a non-default remote).
if [ -f "$stub" ]; then
  REMOTE=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["remote"])' "$stub")
  DEST=$(python   -c 'import json,sys;print(json.load(open(sys.argv[1]))["dest"])'   "$stub")
fi
name="$RUN_ID.tar.zst"
remote_path="$REMOTE:$DEST/$name"

if [ -d "$dir" ] && [ "$FORCE" -ne 1 ]; then
  echo "local $dir already exists — pass --force to overwrite" >&2; exit 1
fi
rclone lsf "$remote_path" >/dev/null 2>&1 || { echo "remote archive not found: $remote_path" >&2; exit 1; }

echo "==> restoring $RUN_ID from $remote_path"
[ "$FORCE" -eq 1 ] && rm -rf "$dir"
rclone cat "$remote_path" | zstd -dc | tar -C "$OUTPUTS" -xf -

[ -d "$dir" ] || { echo "ERROR: expected $dir after extract — not found" >&2; exit 1; }
echo "   restored: $(du -sh "$dir" | cut -f1)  ($(find "$dir" -type f | wc -l) files)"
rm -f "$stub"

if [ "$PURGE" -eq 1 ]; then
  rclone deletefile "$remote_path"
  echo "   purged remote archive $remote_path"
fi
echo "done."
