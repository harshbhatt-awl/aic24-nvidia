from __future__ import annotations

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
    """First eligibility gate the track fails, mirroring create_camera_dict.

    Order: untracked (OfflineID<0) -> length (len(all_serials)<short_track_th)
    -> keypoint (score>keypoint_condition_th) -> kept.
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
    linked: dict[tuple[str, int], bool] = defaultdict(bool)
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
        tracks.append(Track(cam, oid, n, n_all, score, linked[(cam, oid)]))
    return tracks
