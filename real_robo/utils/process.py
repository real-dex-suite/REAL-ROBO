"""Process helpers for the lightweight REAL-ROBO framework.

Each helper returns ``multiprocessing.Process`` objects so that the existing
launch scripts (see ``tools/teleoperation/``) can continue to orchestrate data
collection pipelines while we progressively migrate away from the legacy
dependencies.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from termcolor import cprint

from real_robo.device.camera.realsense import RealSenseRGBDCamera
from real_robo.robots.franka_env_wrapper import FrankaEnvWrapper

try:  # Hydra/OmegaConf is optional at runtime but used in our configs
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - fallback when OmegaConf is absent
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def notify_process_start(message: str) -> None:
    """Print a formatted banner that a subprocess has started."""

    cprint("***************************************************************", "green")
    cprint(f"     {message}", "green")
    cprint("***************************************************************", "green")


def _to_container(config: Any) -> Dict[str, Any]:
    """Convert Hydra configs into plain dictionaries."""

    if OmegaConf is not None and isinstance(config, DictConfig):  # pragma: no cover - depends on Hydra
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, dict):
        return dict(config)
    # Fallback: expose attributes when possible
    result: Dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except AttributeError:  # pragma: no cover - defensive
            continue
        if callable(value):
            continue
        result[name] = value
    return result


def _ensure_path(path_like: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Camera streaming helpers
# ---------------------------------------------------------------------------

@dataclass
class CameraStreamOptions:
    serial: str
    camera_id: int
    frame_rate: int = 30
    align: bool = True
    output_dir: Optional[Path] = None
    visualize: bool = False
    report_interval: float = 1.0  # seconds


def _camera_stream_worker(options: CameraStreamOptions) -> None:
    notify_process_start(f"Starting RealSense camera #{options.camera_id}")
    last_report = 0.0
    camera = RealSenseRGBDCamera(
        serial=options.serial,
        frame_rate=options.frame_rate,
        align=options.align,
    )

    save_dir = options.output_dir
    if save_dir is not None:
        save_dir = save_dir / f"camera_{options.camera_id:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            rgb, depth = camera.get_rgbd_image()
            timestamp = time.time()

            if save_dir is not None:
                file_path = save_dir / f"frame_{timestamp:.3f}.npz"
                np.savez_compressed(file_path, color=rgb, depth=depth, timestamp=timestamp)

            if options.visualize and timestamp - last_report >= options.report_interval:
                cprint(
                    f"[Camera {options.camera_id}] rgb={rgb.shape} depth={depth.shape} t={timestamp:.2f}",
                    "cyan",
                )
                last_report = timestamp
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        cprint(f"Camera stream {options.camera_id} interrupted", "yellow")


def start_robot_cam_stream(
    cam_serial_num: str,
    robot_cam_num: int,
    *,
    frame_rate: int = 30,
    output_dir: Optional[Union[str, Path]] = None,
    visualize: bool = False,
) -> Process:
    """Return a process that streams a RealSense camera to disk."""

    options = CameraStreamOptions(
        serial=cam_serial_num,
        camera_id=robot_cam_num,
        frame_rate=frame_rate,
        output_dir=_ensure_path(output_dir),
        visualize=visualize,
    )
    proc = Process(target=_camera_stream_worker, args=(options,), daemon=False)
    return proc


def get_camera_stream_processes(configs: Any) -> Tuple[List[Process], List[Process]]:
    """Create camera streaming processes based on a Hydra-style config."""

    cfg = _to_container(configs)
    serials: Sequence[str] = cfg.get("robot_cam_serial_numbers", []) or []
    frame_rate = int(cfg.get("frame_rate", cfg.get("camera_frame_rate", 30)))
    visualize = bool(cfg.get("visualize_stream", False))
    output_dir = cfg.get("camera_output_dir")

    camera_processes: List[Process] = []
    for idx, serial in enumerate(serials):
        camera_processes.append(
            start_robot_cam_stream(
                serial,
                idx + 1,
                frame_rate=frame_rate,
                output_dir=output_dir,
                visualize=visualize,
            )
        )

    # We no longer provide a separate TCP streaming helper; return an empty list
    return camera_processes, []


