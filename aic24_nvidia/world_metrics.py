from __future__ import annotations
import sys
from pathlib import Path

import numpy as np


def world_similarity(gt_dets: np.ndarray, pred_dets: np.ndarray, d_max: float) -> np.ndarray:
    """Distance-gated similarity in [0,1]: max(0, 1 - euclidean/d_max).

    gt_dets: (n_gt, 2), pred_dets: (n_pred, 2). Returns (n_gt, n_pred).
    """
    if len(gt_dets) == 0 or len(pred_dets) == 0:
        return np.zeros((len(gt_dets), len(pred_dets)), dtype=float)
    diff = gt_dets[:, None, :] - pred_dets[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    sim = 1.0 - dist / float(d_max)
    return np.clip(sim, 0.0, 1.0)


def load_world_txt(path: Path) -> dict[int, tuple[list[int], np.ndarray]]:
    """Read `frame,id,x,y` rows into {frame: (ids, (N,2) array)}."""
    per_frame: dict[int, tuple[list[int], list[list[float]]]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f, oid, x, y = line.split(",")[:4]
        frame = int(float(f))
        ids, dets = per_frame.setdefault(frame, ([], []))
        ids.append(int(float(oid)))
        dets.append([float(x), float(y)])
    return {f: (ids, np.array(dets, dtype=float)) for f, (ids, dets) in per_frame.items()}


def run_world_eval(gt_txt: Path, pred_txt: Path, d_max: float,
                   trackeval_root: Path, seq_name: str = "scene_001") -> dict:
    """Run TrackEval HOTA + Identity on world-coordinate tracks.

    Defines a minimal _BaseDataset subclass inline (after putting TrackEval on
    sys.path) so this module imports fine without TrackEval installed.
    """
    trackeval_root = Path(trackeval_root)
    if str(trackeval_root) not in sys.path:
        sys.path.insert(0, str(trackeval_root))
    import trackeval  # noqa: E402
    from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402

    gt_frames = load_world_txt(gt_txt)
    pred_frames = load_world_txt(pred_txt) if Path(pred_txt).exists() else {}
    all_frames = sorted(set(gt_frames) | set(pred_frames))

    class _WorldDataset(_BaseDataset):
        def __init__(self):
            super().__init__()
            self.should_classes_combine = False
            self.use_super_categories = False
            self.class_list = ["pedestrian"]
            self.seq_list = [seq_name]
            self.output_fol = None
            self.output_sub_fol = None

        def get_name(self):
            return "NvidiaMTMCWorld"

        def get_display_name(self, tracker):
            return tracker

        def _load_raw_file(self, tracker, seq, is_gt):
            src = gt_frames if is_gt else pred_frames
            n = len(all_frames)
            id_key = "gt_ids" if is_gt else "tracker_ids"
            det_key = "gt_dets" if is_gt else "tracker_dets"
            raw = {id_key: [], det_key: []}
            for fr in all_frames:
                ids, dets = src.get(fr, ([], np.empty((0, 2))))
                raw[id_key].append(np.array(ids, dtype=int))
                raw[det_key].append(np.asarray(dets, dtype=float))
            raw["num_timesteps"] = n
            if not is_gt:
                raw["tracker_confidences"] = [np.ones(len(x)) for x in raw["tracker_ids"]]
            return raw

        def get_preprocessed_seq_data(self, raw_data, cls):
            data = {
                "num_timesteps": raw_data["num_timesteps"],
                "gt_ids": raw_data["gt_ids"],
                "tracker_ids": raw_data["tracker_ids"],
                "gt_dets": raw_data["gt_dets"],
                "tracker_dets": raw_data["tracker_dets"],
                "similarity_scores": raw_data["similarity_scores"],
                "tracker_confidences": raw_data["tracker_confidences"],
            }
            unique_gt = (np.unique(np.concatenate(data["gt_ids"]))
                         if any(len(x) for x in data["gt_ids"]) else np.array([]))
            unique_tr = (np.unique(np.concatenate(data["tracker_ids"]))
                         if any(len(x) for x in data["tracker_ids"]) else np.array([]))
            data["num_gt_dets"] = int(sum(len(x) for x in data["gt_ids"]))
            data["num_tracker_dets"] = int(sum(len(x) for x in data["tracker_ids"]))
            self._remap(data, "gt_ids", unique_gt)
            self._remap(data, "tracker_ids", unique_tr)
            data["num_gt_ids"] = len(unique_gt)
            data["num_tracker_ids"] = len(unique_tr)
            return data

        @staticmethod
        def _remap(data, key, unique):
            lut = {old: new for new, old in enumerate(unique)}
            data[key] = [np.array([lut[i] for i in arr], dtype=int) for arr in data[key]]

        def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
            return world_similarity(np.asarray(gt_dets_t), np.asarray(tracker_dets_t), d_max)

    evaluator = trackeval.Evaluator({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False, "PRINT_ONLY_COMBINED": True,
    })
    dataset = _WorldDataset()
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.Identity(), trackeval.metrics.CLEAR()]
    res, _ = evaluator.evaluate([dataset], metrics)
    seq = res["NvidiaMTMCWorld"]["world"][seq_name]["pedestrian"]
    hota = seq["HOTA"]
    ident = seq["Identity"]
    clear = seq["CLEAR"]
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "IDF1": float(ident["IDF1"]),
        "MOTA": float(clear["MOTA"]),
        "d_max_m": float(d_max),
    }
