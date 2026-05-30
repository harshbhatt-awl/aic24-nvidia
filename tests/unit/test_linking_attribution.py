from aic24_nvidia.diagnostics.linking_attribution import (
    Track, classify_track, G0_UNTRACKED, E1_SHORT, E2_KEYPOINT, KEPT,
)


def _t(offline_id=5, n_all_serials=200, score=1, n_detections=10, linked=True):
    return Track(camera="395", offline_id=offline_id, n_detections=n_detections,
                 n_all_serials=n_all_serials, score=score, linked=linked)


def test_classify_kept_when_long_and_good_keypoints():
    assert classify_track(_t(n_all_serials=200, score=1), short_track_th=120,
                          keypoint_condition_th=1) == KEPT


def test_classify_e2_when_keypoint_score_exceeds_threshold():
    assert classify_track(_t(n_all_serials=200, score=2), short_track_th=120,
                          keypoint_condition_th=1) == E2_KEYPOINT


def test_classify_e1_when_track_too_short():
    assert classify_track(_t(n_all_serials=50, score=1), short_track_th=120,
                          keypoint_condition_th=1) == E1_SHORT


def test_classify_e1_takes_precedence_over_e2():
    # fails both gates; length is checked first
    assert classify_track(_t(n_all_serials=50, score=4), short_track_th=120,
                          keypoint_condition_th=1) == E1_SHORT


def test_classify_g0_when_untracked():
    assert classify_track(_t(offline_id=-1, n_all_serials=-1, score=-1),
                          short_track_th=120, keypoint_condition_th=1) == G0_UNTRACKED


import json
from aic24_nvidia.diagnostics.linking_attribution import load_tracks


def _write(tmp_path, whole, rep):
    (tmp_path / "whole_tracking_results.json").write_text(json.dumps(whole))
    (tmp_path / "representative_nodes_scene1.json").write_text(json.dumps(rep))
    return (tmp_path / "whole_tracking_results.json",
            tmp_path / "representative_nodes_scene1.json")


def test_load_tracks_joins_counts_scores_and_linked(tmp_path):
    whole = {
        "395": {
            "00000000": {"OfflineID": 7, "GlobalOfflineID": 14},
            "00000001": {"OfflineID": 7, "GlobalOfflineID": 14},
            "00000002": {"OfflineID": 8, "GlobalOfflineID": None},
            "00000003": {"OfflineID": -1, "GlobalOfflineID": None},
        }
    }
    rep = {
        "395": {
            "7": {"representative_node": {"score": 1}, "all_serials": ["a"] * 200},
            "8": {"representative_node": {"score": 3}, "all_serials": ["a"] * 50},
        }
    }
    wj, rj = _write(tmp_path, whole, rep)
    tracks = {(t.camera, t.offline_id): t for t in load_tracks(wj, rj)}

    assert tracks[("395", 7)].n_detections == 2
    assert tracks[("395", 7)].n_all_serials == 200
    assert tracks[("395", 7)].score == 1
    assert tracks[("395", 7)].linked is True

    assert tracks[("395", 8)].n_detections == 1
    assert tracks[("395", 8)].n_all_serials == 50
    assert tracks[("395", 8)].linked is False

    # untracked detection: offline_id normalized to -1, no rep node -> (-1, -1)
    assert tracks[("395", -1)].n_detections == 1
    assert tracks[("395", -1)].n_all_serials == -1
    assert tracks[("395", -1)].score == -1


from aic24_nvidia.diagnostics.linking_attribution import attribute, reconcile


def _tracks():
    # two cameras; counts chosen so totals are easy to check
    return [
        Track("390", 1, n_detections=100, n_all_serials=200, score=1, linked=True),   # kept
        Track("390", 2, n_detections=30,  n_all_serials=50,  score=1, linked=False),  # E1
        Track("390", 3, n_detections=40,  n_all_serials=200, score=3, linked=False),  # E2
        Track("395", 4, n_detections=10,  n_all_serials=-1,  score=-1, linked=False), # E1 (absent rep)
        Track("395", -1, n_detections=5,  n_all_serials=-1,  score=-1, linked=False), # G0
    ]


def test_attribute_rolls_up_per_camera_and_gate():
    per_cam, totals = attribute(_tracks(), short_track_th=120, keypoint_condition_th=1)
    assert per_cam["390"][KEPT] == 100
    assert per_cam["390"][E1_SHORT] == 30
    assert per_cam["390"][E2_KEYPOINT] == 40
    assert per_cam["395"][E1_SHORT] == 10
    assert per_cam["395"][G0_UNTRACKED] == 5
    assert totals[KEPT] == 100
    assert totals[E1_SHORT] == 40
    assert totals[E2_KEYPOINT] == 40
    assert totals[G0_UNTRACKED] == 5


def test_reconcile_flags_clean_when_kept_matches_linked():
    rep = reconcile(_tracks(), short_track_th=120, keypoint_condition_th=1)
    assert rep["dropped"] == 85          # 30+40+10+5
    assert rep["kept"] == 100
    assert rep["kept_linked_mismatch"] == 0   # classify==KEPT iff linked, on this fixture


from aic24_nvidia.diagnostics.linking_attribution import format_table


def test_format_table_contains_gate_columns_and_totals():
    per_cam, totals = attribute(_tracks(), short_track_th=120, keypoint_condition_th=1)
    text = format_table(per_cam, totals)
    assert "E2_keypoint" in text
    assert "E1_short_track" in text
    assert "ALL" in text
    # the kept total (100) and an E1 total (40) appear in the rendered table
    assert "100" in text and "40" in text
