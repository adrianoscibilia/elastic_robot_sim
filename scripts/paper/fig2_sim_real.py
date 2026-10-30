"""Fig. 2 with photos: simulated and real platforms side by side.

Panels, left to right: (a) simulated iiwa, (b) real iiwa (bare flange), (c)
simulated UR10, (d) real UR10.  The simulated panels are crops of the R-5
single renders (``fig_robots_sim_{iiwa,ur10}.png``, fig2_renders.py); the
photos are cropped to the same portrait aspect so that each robot fills about
the same fraction of its panel height.  The annotated variant adds the joint
labels on the simulated panels, at the joint anchors projected by
fig2_renders.py (recovered from fig_robots_sim_annotated, render pixels).

    python fig2_sim_real.py [--fig-dir ~/Projects/lnn_paper/figures/v2]

Inputs in --fig-dir: fig_robots_sim_iiwa.png, fig_robots_sim_ur10.png,
photos/iiwa_real_bare_flange.png (remove_gripper.py), photos/ur10_real.jpg.
Outputs in --fig-dir: fig_robots_sim_real.{pdf,png}, fig_robots_sim_real_annotated.{pdf,png}.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

PAPER = Path(os.environ.get("PAPER_ROOT", Path.home() / "Projects" / "lnn_paper"))

ASPECT = 1.32            # panel height / width
PANEL_PX = 750           # panel width in pixels after downsampling
# Crop boxes (x0, y0, x1, y1) in source pixels, all with h/w = ASPECT and the robot
# filling ~91 % of the panel height (top margin ~6 %).
CROPS = {
    "iiwa_sim": (565, 29, 1434, 1176),
    "iiwa_real": (0, 13, 1932, 2563),
    "ur10_sim": (655, 116, 1354, 1039),
    "ur10_real": (176, 378, 1024, 1498),
}
# Joint anchors in render pixels (2000 x 1500), from fig2_renders.project().
ANCHORS = {
    "iiwa_sim": {"A1": (937.9, 961.7), "A2": (926.6, 767.2), "A3": (945.1, 689.1), "A4": (961.8, 618.4),
                 "A5": (1100.1, 422.9), "A6": (1183.3, 195.9), "A7": (1309.5, 158.5)},
    "ur10_sim": {"pan": (1022.8, 904.1), "lift": (1020.4, 904.5), "elbow": (934.4, 672.7),
                 "w1": (1034.9, 226.0), "w2": (1002.5, 249.9), "w3": (1104.1, 267.4)},
}
# Label offsets in points (dx, dy; dy > 0 = up).
OFFSETS = {
    "iiwa_sim": {"A1": (-16, 4), "A2": (-17, -2), "A3": (14, -6), "A4": (16, 2), "A5": (12, 6),
                 "A6": (-16, -4), "A7": (6, 10)},
    "ur10_sim": {"pan": (-15, -4), "lift": (14, -2), "elbow": (-16, 6), "w1": (-22, 10), "w2": (-40, -4),
                 "w3": (10, -8)},
}
GROUPS = [("KUKA LBR iiwa 14 R820", ("iiwa_sim", "iiwa_real")), ("UR10 CB3", ("ur10_sim", "ur10_real"))]
SUBLABELS = {"iiwa_sim": "(a) simulated", "iiwa_real": "(b) real", "ur10_sim": "(c) simulated", "ur10_real": "(d) real"}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return plt


def load_panels(fig_dir: Path):
    src = {
        "iiwa_sim": fig_dir / "fig_robots_sim_iiwa.png",
        "iiwa_real": fig_dir / "photos" / "iiwa_real_bare_flange.png",
        "ur10_sim": fig_dir / "fig_robots_sim_ur10.png",
        "ur10_real": fig_dir / "photos" / "ur10_real.jpg",
    }
    panels = {}
    for key, path in src.items():
        img = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = CROPS[key]
        assert abs((y1 - y0) / (x1 - x0) - ASPECT) < 0.01, (key, (y1 - y0) / (x1 - x0))
        img = img.crop((x0, y0, x1, y1))
        scale = min(1.0, PANEL_PX / img.width)            # ~430 dpi at the printed panel width
        panels[key] = (np.asarray(img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)),
                       scale)
    return panels


def compose(panels, annotated: bool, out: Path):
    plt = setup_matplotlib()
    W = 7.16                                    # IEEE two-column width (in)
    gap_in, gap_out = 0.04, 0.14                # inside a robot pair / between the pairs
    pw = (W - 2 * gap_in - gap_out) / 4
    ph = pw * ASPECT
    top, bottom = 0.20, 0.17                    # group titles / panel sub-labels
    H = top + ph + bottom
    fig = plt.figure(figsize=(W, H))
    lefts = [0.0, pw + gap_in, 2 * pw + gap_in + gap_out, 3 * pw + 2 * gap_in + gap_out]
    order = ["iiwa_sim", "iiwa_real", "ur10_sim", "ur10_real"]
    axes = {}
    for key, left in zip(order, lefts):
        ax = fig.add_axes([left / W, bottom / H, pw / W, ph / H])
        img, scale = panels[key]
        ax.imshow(img, interpolation="lanczos")
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.65")
        ax.text(0.5, -0.025, SUBLABELS[key], transform=ax.transAxes, ha="center", va="top", fontsize=8)
        axes[key] = ax
        if annotated and key in ANCHORS:
            x0, y0 = CROPS[key][:2]
            for label, (x, y) in ANCHORS[key].items():
                dx, dy = OFFSETS[key][label]
                ax.annotate(label, xy=((x - x0) * scale, (y - y0) * scale), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                            ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top", color="black",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.5", lw=0.4, alpha=0.9),
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3", shrinkA=0, shrinkB=0))
    # Group titles over each sim/real pair.
    for title, (a, b) in GROUPS:
        l = lefts[order.index(a)]; r = lefts[order.index(b)] + pw
        fig.text((l + r) / 2 / W, (bottom + ph + 0.085) / H, title, ha="center", va="bottom", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", default=str(PAPER / "figures" / "v2"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    fig_dir = Path(args.fig_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else fig_dir
    panels = load_panels(fig_dir)
    compose(panels, False, out_dir / "fig_robots_sim_real")
    compose(panels, True, out_dir / "fig_robots_sim_real_annotated")
    print("wrote", out_dir / "fig_robots_sim_real.{pdf,png}", "and _annotated")


if __name__ == "__main__":
    main()
