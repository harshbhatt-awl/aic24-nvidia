#!/usr/bin/env bash
# Archive a finished pipeline run to OneDrive (or any rclone remote) and free the
# local copy — for when outputs/ is eating your disk.
#
# WHY THIS SHAPE:
#   * A run dir is ~3-4 GB across *tens of thousands* of small files (frames,
#     per-frame detections/embeddings). OneDrive throttles small-file storms
#     (HTTP 429) and a live FUSE mount of it is slow/fragile — so we pack each
#     run into ONE tarball and push that instead of thousands of objects.
#   * We stage the tarball to a temp dir ($AIC24_ARCHIVE_TMP or /tmp — tmpfs on
#     most boxes, i.e. RAM not the full disk) and `rclone copy` it: a real-file
#     copy is chunked, resumable and hash-verified, whereas `rclone rcat`
#     streaming hangs on multi-GB uploads to OneDrive (the upload-session
#     finalize stalls). One run is staged at a time and its temp is deleted right
#     after upload, so no *permanent* local disk is consumed.
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
#   --keep-local    upload but DON'T delete the local copy (backup mode)
#   --no-verify     skip the local `zstd -t` integrity check (faster, less safe)
#   --force         allow archiving "baseline" (NORMALLY REFUSED — experiments/
#                   symlinks its upstream stages from outputs/baseline/)
#   --remote NAME   rclone remote   (default: $AIC24_RCLONE_REMOTE or "onedrive")
#   --dest PATH     remote dir      (default: aic24/outputs-archive)
#   staging dir is $AIC24_ARCHIVE_TMP or $TMPDIR or /tmp (needs room for one tarball)
#
# Restore later with: scripts/restore_run.sh <run_id>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS="$REPO_ROOT/outputs"
PY="python"; [ -x "$REPO_ROOT/.venv/bin/python" ] && PY="$REPO_ROOT/.venv/bin/python"
REMOTE="${AIC24_RCLONE_REMOTE:-onedrive}"
DEST="aic24/outputs-archive"
TMP_ROOT="${AIC24_ARCHIVE_TMP:-${TMPDIR:-/tmp}}"
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

  # Stage the tarball to $TMP_ROOT, then `rclone copy` it. rclone rcat hangs on
  # multi-GB streamed uploads to OneDrive (upload-session finalize stalls); a
  # real-file copy is chunked, resumable and hash-verified. We stage one run at a
  # time and delete the temp right after upload, so no permanent disk is used.
  tmp_tar="$TMP_ROOT/aic24-archive-$run_id.tar.zst"
  avail=$(df -B1 --output=avail "$TMP_ROOT" | tail -1 | tr -d ' ')
  if [ "$avail" -lt "$nbytes" ]; then
    echo "   ERROR: only $(human "$avail") free in $TMP_ROOT, need ~$(human "$nbytes") to stage." >&2
    echo "   Set AIC24_ARCHIVE_TMP to a dir with room and retry." >&2
    exit 1
  fi
  trap 'rm -f "$tmp_tar"' EXIT
  echo "   staging tarball → $tmp_tar (tar | zstd -T0 -3)..."
  tar -C "$OUTPUTS" -cf - "$run_id" | zstd -q -T0 -3 > "$tmp_tar"
  local_bytes=$(stat -c%s "$tmp_tar")
  if [ "$VERIFY" -eq 1 ]; then
    echo "   verifying local archive (zstd -t)..." && zstd -tq "$tmp_tar"
  fi
  echo "   uploading $(human "$local_bytes") (rclone copyto, hash-verified)..."
  rclone copyto "$tmp_tar" "$remote_path" -P

  remote_bytes=$(rclone size --json "$remote_path" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["bytes"])')
  echo "   uploaded: $(human "$remote_bytes")"
  if [ "$remote_bytes" != "$local_bytes" ]; then
    echo "   ERROR: remote size ($remote_bytes) != local ($local_bytes) — NOT deleting local" >&2
    exit 1
  fi
  rm -f "$tmp_tar"; trap - EXIT
  echo "   integrity OK (local zstd -t + hash-verified upload + size match)"

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
