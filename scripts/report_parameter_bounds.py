#!/usr/bin/env python3
"""Generate the bounds checklist that justifies the sampled parameter ranges.

For each joint, reports the declared stiffness interval and its width in
decades, which published anchors (from ``docs/PARAMETER_PROVENANCE.md``)
fall inside it, the predicted transmission-mode frequency range at the
interval's extremes, the control-bandwidth lower bound, and the integration
step the interval's stiff end would require. Writes ``bounds.csv``,
``bounds.md`` and ``modes.png`` to ``--output``.

This turns the justification section of ``R3_04`` into something generated
from the current config rather than typed by hand -- run it again after any
change to ``transmission.stiffness_nominal``/``stiffness_factor`` and diff
the output.

Examples
--------
    python scripts/report_parameter_bounds.py
    python scripts/report_parameter_bounds.py --config config/identification/ur10_table.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim import excitation as exc
from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.dataset import DEFAULT_CONFIG, load_config
from elastic_sim.torque_runners import control_separation_ratio, link_inertia_envelope, link_inertia_max

_STIFFNESS_ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*stiffness\s*\|[^|]*\|[^|]*\|\s*([MPCDE])\s*\|\s*([^|]*)\|")


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO / candidate


def _published_anchors(joint_names: tuple[str, ...]) -> dict[str, str]:
    """One-line summary of the provenance doc's stiffness anchor per joint."""
    doc_path = _REPO / "docs" / "PARAMETER_PROVENANCE.md"
    anchors = {name: "" for name in joint_names}
    if not doc_path.is_file():
        return anchors
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        match = _STIFFNESS_ROW.match(line)
        if not match:
            continue
        joint, letter, source = match.groups()
        if joint in anchors:
            anchors[joint] = f"{letter}: {source.strip()}"
    return anchors


def compute_bounds(config, asset) -> pd.DataFrame:
    """The A6 table: one row per joint, every column independently checkable."""
    names = asset.joint_names
    n = len(names)
    sampling = config.transmission

    if sampling.stiffness_nominal:
        nominal = np.asarray(sampling.stiffness_nominal, dtype=float)
        k_min = nominal * sampling.stiffness_factor[0]
        k_max = nominal * sampling.stiffness_factor[1]
    else:
        intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
        k_min, k_max = intervals[:, 0], intervals[:, 1]

    if sampling.rotor_inertia_nominal:
        rotor = np.asarray(sampling.rotor_inertia_nominal, dtype=float)
    else:
        rotor = np.broadcast_to(np.asarray(sampling.rotor_inertia, dtype=float), (n,))

    bounds = None
    if config.excitation.position_window:
        window_lower, window_upper = exc.effective_position_window(asset, config.excitation)
        bounds = tuple(zip(window_lower.tolist(), window_upper.tolist()))
    link_median, _link_floor = link_inertia_envelope(asset, n_samples=sampling.inertia_samples, bounds=bounds)
    link_max = link_inertia_max(asset, n_samples=sampling.inertia_samples, bounds=bounds)
    j_eff = rotor * link_median / (rotor + link_median)

    f_control = config.control_frequency / (2.0 * np.pi)
    control_bandwidth_bound = (2.0 * np.pi * 5.0 * f_control) ** 2 * j_eff
    mode_min = np.sqrt(k_min / j_eff) / (2.0 * np.pi)
    mode_max = np.sqrt(k_max / j_eff) / (2.0 * np.pi)
    required_step_at_kmax = 1.0 / (20.0 * mode_max)

    # Worst-case control/transmission separation ratio (R4_02 Sec 7,
    # R4_03 Sec 2): softest stiffness, heaviest rotor and link inertia the
    # prior admits, fastest sampled control gain.
    rotor_worst = rotor * sampling.rotor_inertia_factor[1] if sampling.rotor_inertia_nominal else rotor
    j_eff_worst = rotor_worst * link_max / (rotor_worst + link_max)
    omega_worst = (
        config.control_gains.natural_frequency[1] if config.control_gains.enabled else config.control_frequency
    )
    separation_ratio = control_separation_ratio(k_min, rotor_worst, link_max, omega_worst)

    anchors = _published_anchors(names)
    return pd.DataFrame({
        "joint": names,
        "k_min_Nm_per_rad": k_min,
        "k_max_Nm_per_rad": k_max,
        "width_decades": np.log10(k_max / k_min),
        "rotor_inertia_kg_m2": rotor,
        "link_inertia_median_kg_m2": link_median,
        "link_inertia_max_kg_m2": link_max,
        "j_eff_kg_m2": j_eff,
        "mode_min_hz": mode_min,
        "mode_max_hz": mode_max,
        "control_bandwidth_lower_bound_Nm_per_rad": control_bandwidth_bound,
        "lower_bound_satisfied": k_min >= control_bandwidth_bound,
        "required_time_step_at_k_max_s": required_step_at_kmax,
        "control_separation_ratio_worst": separation_ratio,
        "control_separation_satisfied": separation_ratio >= config.control_separation.min_ratio,
        "published_anchor": [anchors[name] for name in names],
    })


def write_reports(bounds: pd.DataFrame, config, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(output / "bounds.csv", index=False)

    lines = [
        "| Joint | k range [Nm/rad] | width [decades] | mode range [Hz] | control bound [Nm/rad] | satisfied "
        "| step @ k_max [s] | separation ratio (worst) | separation ok | anchor |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in bounds.itertuples():
        lines.append(
            f"| {row.joint} | {row.k_min_Nm_per_rad:.0f}-{row.k_max_Nm_per_rad:.0f} "
            f"| {row.width_decades:.2f} | {row.mode_min_hz:.0f}-{row.mode_max_hz:.0f} "
            f"| {row.control_bandwidth_lower_bound_Nm_per_rad:.0f} "
            f"| {'yes' if row.lower_bound_satisfied else 'NO'} "
            f"| {row.required_time_step_at_k_max_s:.2e} "
            f"| {row.control_separation_ratio_worst:.2f} "
            f"| {'yes' if row.control_separation_satisfied else 'NO'} "
            f"| {row.published_anchor} |"
        )
    (output / "bounds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(bounds))
    ax.errorbar(
        x, 0.5 * (bounds["mode_min_hz"] + bounds["mode_max_hz"]),
        yerr=0.5 * (bounds["mode_max_hz"] - bounds["mode_min_hz"]),
        fmt="o", capsize=4, label="declared mode range",
    )
    control_line = 5.0 * config.control_frequency / (2.0 * np.pi)
    nyquist = 0.5 / config.sample_time_step
    ax.axhline(control_line, color="tab:red", linestyle="--", label=f"5x control bandwidth ({control_line:.1f} Hz)")
    ax.axhline(nyquist, color="tab:green", linestyle="--", label=f"output-grid Nyquist ({nyquist:.0f} Hz)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bounds["joint"])
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Transmission mode range vs. control bandwidth and output Nyquist")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "modes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--output", default="reports/parameter_bounds")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    asset = _load_asset(args.asset or config.asset)
    asset.resolve_active_joints()

    bounds = compute_bounds(config, asset)
    output = _resolve(args.output)
    write_reports(bounds, config, output)

    print(bounds.to_string(index=False))
    violated = bounds[~bounds["lower_bound_satisfied"]]
    if not violated.empty:
        print(f"\nwarning: {len(violated)} joint(s) violate the control-bandwidth lower bound: "
              f"{', '.join(violated['joint'])}")
    separation_violated = bounds[~bounds["control_separation_satisfied"]]
    if not separation_violated.empty:
        print(f"\nwarning: {len(separation_violated)} joint(s) fall below control_separation.min_ratio="
              f"{config.control_separation.min_ratio:g} at their worst corner: {', '.join(separation_violated['joint'])}")
    print(f"\nWrote {output / 'bounds.csv'}, {output / 'bounds.md'}, {output / 'modes.png'}")


if __name__ == "__main__":
    main()
