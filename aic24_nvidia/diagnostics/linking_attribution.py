from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Gate labels in attribution (pipeline) order.
G0_UNTRACKED = "G0_untracked"
E1_SHORT = "E1_short_track"
E2_KEYPOINT = "E2_keypoint"
KEPT = "kept"
GATES = (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT, KEPT)


@dataclass(frozen=True)
class Track:
    camera: str           # numeric camera key as in the MCT json, e.g. "395"
    offline_id: int       # SCT local id; -1 means untracked
    n_detections: int     # detections for this (camera, offline_id) in whole_tracking_results
    n_all_serials: int    # rep-node all_serials length (the E1 input); -1 if absent
    score: int            # rep-node representative score (the E2 input); -1 if absent
    linked: bool          # any detection of this track has a non-null GlobalOfflineID


def classify_track(track: Track, short_track_th: int, keypoint_condition_th: int) -> str:
    """First eligibility gate the track fails, in pipeline order.

    E1 (length) and E2 (keypoint) mirror the gates in upstream ``create_camera_dict``.
    G0 (untracked, OfflineID<0) is a diagnostic-added bucket that identifies SCT
    tracks never assigned a global ID; it is not a gate in the upstream code.

    Order: G0 (OfflineID<0) -> E1 (len(all_serials)<short_track_th)
    -> E2 (score>keypoint_condition_th) -> kept.
    """
    if track.offline_id < 0:
        return G0_UNTRACKED
    if track.n_all_serials < short_track_th:
        return E1_SHORT
    if track.score > keypoint_condition_th:
        return E2_KEYPOINT
    return KEPT


def load_tracks(whole_json: str | Path, rep_json: str | Path) -> list[Track]:
    """Build Track records by joining the post-assignment whole_tracking_results
    (per-detection OfflineID/GlobalOfflineID) with representative_nodes (the gate
    inputs all_serials/score) on (camera, offline_id)."""
    whole = json.loads(Path(whole_json).read_text())
    rep = json.loads(Path(rep_json).read_text())

    rep_by: dict[tuple[str, int], tuple[int, int]] = {}
    for cam, nodes in rep.items():
        if not isinstance(nodes, dict):
            continue
        for oid_str, node in nodes.items():
            n_all = len(node["all_serials"])
            score = int(node["representative_node"]["score"])
            rep_by[(cam, int(oid_str))] = (n_all, score)

    ndet: dict[tuple[str, int], int] = defaultdict(int)
    linked: dict[tuple[str, int], bool] = {}
    for cam, entries in whole.items():
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            oid = entry.get("OfflineID")
            oid = -1 if oid is None else int(oid)
            ndet[(cam, oid)] += 1
            if entry.get("GlobalOfflineID") is not None:
                linked[(cam, oid)] = True

    tracks: list[Track] = []
    for (cam, oid), n in ndet.items():
        n_all, score = rep_by.get((cam, oid), (-1, -1))
        tracks.append(Track(cam, oid, n, n_all, score, linked.get((cam, oid), False)))
    return tracks


def attribute(tracks: list[Track], short_track_th: int, keypoint_condition_th: int):
    """Return (per_camera, totals) detection-count rollups keyed by gate label."""
    per_cam: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in tracks:
        gate = classify_track(t, short_track_th, keypoint_condition_th)
        per_cam[t.camera][gate] += t.n_detections
    totals: dict[str, int] = defaultdict(int)
    for gates in per_cam.values():
        for g, c in gates.items():
            totals[g] += c
    return {c: dict(g) for c, g in per_cam.items()}, dict(totals)


def reconcile(tracks: list[Track], short_track_th: int, keypoint_condition_th: int):
    """Roll up and verify the kept<->linked invariant the model rests on."""
    _, totals = attribute(tracks, short_track_th, keypoint_condition_th)
    kept = totals.get(KEPT, 0)
    dropped = sum(totals.get(g, 0) for g in (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT))
    mismatch = 0
    for t in tracks:
        is_kept = classify_track(t, short_track_th, keypoint_condition_th) == KEPT
        if is_kept != t.linked:
            mismatch += t.n_detections
    return {"kept": kept, "dropped": dropped, "totals": totals,
            "kept_linked_mismatch": mismatch}


def format_table(per_cam: dict, totals: dict) -> str:
    cols = [E1_SHORT, E2_KEYPOINT, G0_UNTRACKED, KEPT]
    head = f"{'camera':<8}" + "".join(f"{c:>16}" for c in cols)
    lines = [head, "-" * len(head)]
    for cam in sorted(per_cam):
        row = f"{cam:<8}" + "".join(f"{per_cam[cam].get(c, 0):>16}" for c in cols)
        lines.append(row)
    lines.append("-" * len(head))
    lines.append(f"{'ALL':<8}" + "".join(f"{totals.get(c, 0):>16}" for c in cols))
    dropped = sum(totals.get(c, 0) for c in (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT))
    lines.append(f"\ndropped (G0+E1+E2) = {dropped} | kept = {totals.get(KEPT, 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attribute MCT-dropped detections to eligibility gates.")
    p.add_argument("--mct-dir", required=True,
                   help="path to outputs/<run>/mct/scene_001 (holds whole_tracking_results.json + representative_nodes_scene1.json)")
    p.add_argument("--short-track-th", type=int, default=120)
    p.add_argument("--keypoint-condition-th", type=int, default=1)
    args = p.parse_args(argv)

    mct = Path(args.mct_dir)
    tracks = load_tracks(mct / "whole_tracking_results.json",
                         mct / "representative_nodes_scene1.json")
    per_cam, totals = attribute(tracks, args.short_track_th, args.keypoint_condition_th)
    rep = reconcile(tracks, args.short_track_th, args.keypoint_condition_th)
    print(format_table(per_cam, totals))
    if rep["kept_linked_mismatch"]:
        print(f"WARNING: kept/linked mismatch on {rep['kept_linked_mismatch']} detections "
              "(the kept<->eligible invariant did not hold for this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
