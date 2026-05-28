from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import yaml
from .errors import ConfigError


@dataclass(frozen=True)
class ClipCfg:
    start_sec: float
    duration_sec: float


@dataclass(frozen=True)
class DetectCfg:
    conf_thresh: float
    nms_iou: float


@dataclass(frozen=True)
class ReidCfg:
    similarity_thresh: float


@dataclass(frozen=True)
class PoseCfg:
    keypoint_conf: float


@dataclass(frozen=True)
class SctCfg:
    track_buffer: int
    match_thresh: float


@dataclass(frozen=True)
class MctCfg:
    cluster_thresh: float
    min_track_len: int
    hard_world_gate: bool = False


@dataclass(frozen=True)
class EvalCfg:
    world_d_max: float = 1.0


@dataclass(frozen=True)
class WorldProjectionCfg:
    method: str = "bbox_bottom"        # bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback
    ankle_min_conf: float = 0.3


@dataclass(frozen=True)
class Config:
    scene: str
    data_root: Path
    weights_root: Path
    outputs_root: Path
    external_root: Path
    clip: ClipCfg
    detect: DetectCfg
    reid: ReidCfg
    pose: PoseCfg
    sct: SctCfg
    mct: MctCfg
    eval: EvalCfg
    world_projection: WorldProjectionCfg
    tracking_params: Mapping[str, object]
    vram_min_free_gb: float
    fps: int
    config_path: Path

    @property
    def config_filename(self) -> str:
        return self.config_path.stem

    @property
    def scene_dir(self) -> Path:
        return self.data_root / "nvidia_mtmc_2024" / self.scene

    @property
    def yachiyo_root(self) -> Path:
        return self.external_root / "AIC24_Track1_YACHIYO_RIIPS"


REQUIRED = {
    "scene": str,
    "data_root": str,
    "weights_root": str,
    "outputs_root": str,
    "external_root": str,
    "clip": dict,
    "detect": dict,
    "reid": dict,
    "pose": dict,
    "sct": dict,
    "mct": dict,
    "vram_min_free_gb": (int, float),
    "fps": int,
}


def load_config(path: Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open() as f:
        body = yaml.safe_load(f)
    if not isinstance(body, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    for key, types in REQUIRED.items():
        if key not in body:
            raise ConfigError(f"missing required field: {key}")
        if not isinstance(body[key], types):
            raise ConfigError(f"field '{key}' must be {types}, got {type(body[key])}")

    clip = ClipCfg(**body["clip"])
    if clip.duration_sec <= 0:
        raise ConfigError("clip.duration_sec must be > 0")
    if clip.start_sec < 0:
        raise ConfigError("clip.start_sec must be >= 0")

    mct = MctCfg(
        cluster_thresh=body["mct"]["cluster_thresh"],
        min_track_len=body["mct"]["min_track_len"],
        hard_world_gate=bool(body["mct"].get("hard_world_gate", False)),
    )
    eval_body = body.get("eval") or {}
    eval_cfg = EvalCfg(**eval_body) if eval_body else EvalCfg()
    wp_body = body.get("world_projection") or {}
    wp_method = wp_body.get("method", "bbox_bottom")
    if wp_method not in {"bbox_bottom", "ankle_avg", "ankle_lower", "ankle_w_fallback"}:
        raise ConfigError(f"world_projection.method must be one of bbox_bottom|ankle_avg|ankle_lower|ankle_w_fallback, got {wp_method!r}")
    wp_min_conf = float(wp_body.get("ankle_min_conf", 0.3))
    if not (0.0 <= wp_min_conf <= 1.0):
        raise ConfigError(f"world_projection.ankle_min_conf must be in [0, 1], got {wp_min_conf}")
    world_projection = WorldProjectionCfg(method=wp_method, ankle_min_conf=wp_min_conf)
    tracking_params = MappingProxyType(dict(body.get("tracking_params") or {}))

    return Config(
        scene=body["scene"],
        data_root=Path(body["data_root"]).resolve(),
        weights_root=Path(body["weights_root"]).resolve(),
        outputs_root=Path(body["outputs_root"]).resolve(),
        external_root=Path(body["external_root"]).resolve(),
        clip=clip,
        detect=DetectCfg(**body["detect"]),
        reid=ReidCfg(**body["reid"]),
        pose=PoseCfg(**body["pose"]),
        sct=SctCfg(**body["sct"]),
        mct=mct,
        eval=eval_cfg,
        world_projection=world_projection,
        tracking_params=tracking_params,
        vram_min_free_gb=float(body["vram_min_free_gb"]),
        fps=int(body["fps"]),
        config_path=path.resolve(),
    )
