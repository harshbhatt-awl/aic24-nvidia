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
