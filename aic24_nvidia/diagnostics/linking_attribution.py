from __future__ import annotations

from dataclasses import dataclass

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
