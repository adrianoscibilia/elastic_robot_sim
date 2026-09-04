"""Simulator-independent plots for normalized experiment rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from .assets import AssetSpec
from .kinematics import kinematic_groups


def plot_rollouts(
    frames: Mapping[str, pd.DataFrame], asset: AssetSpec, *, output_dir: str | Path | None = None,
    show: bool = True,
) -> tuple[Path, ...]:
    """Plot joint and Cartesian signals, optionally persisting PNG and SVG."""
    if not frames:
        return ()
    saved: list[Path] = []
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    groups = kinematic_groups(asset)
    for group in groups:
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9), constrained_layout=True)
        figure.suptitle(f"{asset.name}: {group.name}")
        for backend, frame in frames.items():
            for joint in group.joints:
                if f"q_ref__{joint}" in frame:
                    axes[0].plot(frame.t, frame[f"q_ref__{joint}"], "--", alpha=0.55, label=f"{backend} {joint} ref")
                if f"q__{joint}" in frame:
                    axes[0].plot(frame.t, frame[f"q__{joint}"], label=f"{backend} {joint}")
                if f"q_motor__{joint}" in frame:
                    axes[0].plot(frame.t, frame[f"q_motor__{joint}"], ":", alpha=0.75, label=f"{backend} {joint} motor")
                if f"dq__{joint}" in frame:
                    axes[1].plot(frame.t, frame[f"dq__{joint}"], label=f"{backend} {joint}")
                if f"dq_motor__{joint}" in frame:
                    axes[1].plot(frame.t, frame[f"dq_motor__{joint}"], ":", alpha=0.75, label=f"{backend} {joint} motor")
                if f"tau_motor__{joint}" in frame:
                    axes[2].plot(frame.t, frame[f"tau_motor__{joint}"], label=f"{backend} {joint}")
                if f"tau_link__{joint}" in frame:
                    axes[2].plot(frame.t, frame[f"tau_link__{joint}"], "--", alpha=0.75, label=f"{backend} {joint} link")
        axes[0].set_ylabel("position [rad or m]")
        axes[1].set_ylabel("velocity")
        axes[2].set_ylabel("effort")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize="x-small", ncols=2)
        saved.extend(_save(figure, destination, f"joints_{group.name}"))

        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8), constrained_layout=True)
        figure.suptitle(f"{asset.name}: {group.name} Cartesian tracking")
        for backend, frame in frames.items():
            for coordinate in "xyz":
                ref, measured = f"ee_ref__{group.name}__{coordinate}", f"ee__{group.name}__{coordinate}"
                if ref in frame:
                    axes[0].plot(frame.t, frame[ref], "--", alpha=0.55, label=f"{backend} {coordinate} ref")
                if measured in frame:
                    axes[0].plot(frame.t, frame[measured], label=f"{backend} {coordinate}")
            error = f"ee_position_error__{group.name}"
            if error in frame:
                axes[1].plot(frame.t, frame[error], label=f"{backend} position")
            orientation = f"ee_orientation_error__{group.name}"
            if orientation in frame:
                axes[1].plot(frame.t, frame[orientation], label=f"{backend} orientation")
            if "self_collision_clearance" in frame:
                axes[2].plot(frame.t, frame.self_collision_clearance, label=backend)
        axes[0].set_ylabel("position [m]")
        axes[1].set_ylabel("tracking error")
        axes[2].set_ylabel("clearance [m]")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize="small")
        saved.extend(_save(figure, destination, f"cartesian_{group.name}"))
    wrench_prefixes = (
        ("force_flange", "flange force"), ("force_link_side", "link-side force"),
        ("torque_flange", "flange torque"), ("torque_link_side", "link-side torque"),
    )
    if any(any(f"{prefix}__{axis}" in frame for prefix, _ in wrench_prefixes for axis in "xyz") for frame in frames.values()):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7), constrained_layout=True)
        figure.suptitle(f"{asset.name}: Cartesian wrench")
        for backend, frame in frames.items():
            for prefix, label in wrench_prefixes:
                target_axis = axes[0] if prefix.startswith("force") else axes[1]
                for axis in "xyz":
                    column = f"{prefix}__{axis}"
                    if column in frame:
                        target_axis.plot(frame.t, frame[column], label=f"{backend} {label} {axis}")
        axes[0].set_ylabel("force [N]")
        axes[1].set_ylabel("torque [Nm]")
        axes[1].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize="x-small", ncols=2)
        saved.extend(_save(figure, destination, "cartesian_wrench"))
    if show:
        plt.show()
    else:
        plt.close("all")
    return tuple(saved)


def _save(figure: object, destination: Path | None, stem: str) -> list[Path]:
    if destination is None:
        return []
    paths = [destination / f"{stem}.png", destination / f"{stem}.svg"]
    for path in paths:
        figure.savefig(path, dpi=160)
    return paths
