import numpy as np
from aic24_nvidia.world_metrics import world_similarity, load_world_txt


def test_world_similarity_distance_gate():
    gt = np.array([[0.0, 0.0], [10.0, 10.0]])      # 2 gt points
    pred = np.array([[0.5, 0.0]])                   # 1 pred, 0.5 m from gt[0]
    sim = world_similarity(gt, pred, d_max=1.0)     # shape (n_gt, n_pred)
    assert sim.shape == (2, 1)
    assert abs(sim[0, 0] - 0.5) < 1e-6              # 1 - 0.5/1.0
    assert sim[1, 0] == 0.0                          # >1 m -> gated to 0


def test_load_world_txt(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("1,7,3.0,6.0\n1,8,1.0,1.0\n2,7,1.5,1.5\n")
    per_frame = load_world_txt(p)
    assert set(per_frame.keys()) == {1, 2}
    ids, dets = per_frame[1]
    assert ids == [7, 8]
    assert dets.tolist() == [[3.0, 6.0], [1.0, 1.0]]
