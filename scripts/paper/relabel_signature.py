"""Fig. 6 (fig_iiwa_transmission_signature) in the paper style, relabelled "2M-KALNN" (A2L_013 F-b, A2L_015 F-24).

Runs ``fig2()`` of ``REFACTOR_SPECS/final_results/tools/make_real_figures.py``
(the figure's generator, left untouched) from its source with its output
directory redirected:

1. with only the A2L_013 legend relabel, checking that it reproduces
   ``REFACTOR_SPECS/paper_v2/assets/figs/fig_iiwa_transmission_signature.png``
   (the L2A_011 figure), so the environment draws the same figure;
2. with the relabel plus the paper style of ``common.setup_matplotlib`` on a
   white background, "(a)"–"(c)" panel titles, no data footer and a 7.16 in
   width at the generator's aspect ratio, written as PDF and PNG to
   ``~/Projects/lnn_paper/figures/v2/``.

    ~/Projects/dynamic_model_nn/.venv/bin/python scripts/paper/relabel_signature.py
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

import common as C

GENERATOR = C.ERS / "REFACTOR_SPECS/final_results/tools/make_real_figures.py"
NAME = "fig_iiwa_transmission_signature"
REFERENCE = C.ERS / "REFACTOR_SPECS/paper_v2/assets/figs" / f"{NAME}.png"
OUT = C.FIG_DIR
GENERATED = "real_fig2_iiwa_transmission_signature"
RELABEL = [('"KALNN two-mass, filter bypassed"', '"2M-KALNN, filter bypassed"'),
           ('("kalnn_twomass", 1, "on", "KALNN two-mass", BLUE, "")', '("kalnn_twomass", 1, "on", "2M-KALNN", BLUE, "")')]
WIDTH = 7.16
SCALE = WIDTH / 13.2  # the generator's figure is 13.2 x 4.2 in
COMPACT = "handlelength=1.4, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2"
STYLE = [
    ('figsize=(13.2, 4.2)', f'figsize=({WIDTH}, {4.2 * SCALE:.4f})'),
    ('ax.set_title("a. measured transfer on the real iiwa")', 'ax.set_title("(a) measured transfer on the real iiwa")'),
    ('"b. iiwa A2, slow file"', '"(b) iiwa A2, slow file"'),
    ('"c. iiwa A2, fast file"', '"(c) iiwa A2, fast file"'),
    # >= 7 pt everywhere (paper rule): the generator sets 7.5 / 6.8 pt explicitly
    ('fontsize=7.5)', 'fontsize=7)', 3),  # one of them is in fig1(), not drawn here
    # 7 pt legends on a 7.16 in figure: compact handles and spacing so they stay clear of the data
    # (a) with the axis taken down to 1e-2 so the legend sits below the lines falling at 15-40 Hz,
    # (b) with the long "measured torque" entry on two lines so it stays left of the 40-500 Hz bars
    ('ax.legend(fontsize=6.8, ncol=2, loc="lower left")',
     f'ax.set_ylim(bottom=0.01); ax.legend(fontsize=7, ncol=2, loc="lower left", {COMPACT})'),
    ('ax.legend(fontsize=6.8, loc="upper left")', f'ax.legend(fontsize=7, loc="upper left", {COMPACT})'),
    ('label="measured torque RMS in band")', 'label="measured torque\\nRMS in band")'),
    # the 9-word axis label does not fit a 1.8 in tall axis at 8 pt
    ('"|H| = measured torque / rigid torque from motor state"', '"|H| = measured / rigid torque"'),
    # line and marker sizes scaled with the width, so they keep their proportion to the panels
    ('ms=5, lw=1.6,', 'ms=3.2, lw=1.0,'),
    ('ax.axhline(1.0, color=INK2, lw=1, ls=":")', 'ax.axhline(1.0, color=INK2, lw=0.7, ls=":")'),
    ('s=420, color=VIOLET, zorder=4, linewidths=2,', 's=130, color=VIOLET, zorder=4, linewidths=1.3,'),
    ('fig.text(0.01, 0.01, "Welch spectra', 'False and fig.text(0.01, 0.01, "Welch spectra'),
    ('fig.tight_layout(rect=(0, 0.03, 1, 1))', 'fig.tight_layout(pad=0.3, w_pad=1.0)', 2),  # also in fig3()
    ('fig.savefig(FIG / "real_fig2_iiwa_transmission_signature.png", dpi=200)',
     'fig.savefig(FIG / "real_fig2_iiwa_transmission_signature.pdf"); '
     'fig.savefig(FIG / "real_fig2_iiwa_transmission_signature.png", dpi=300)'),
]


def edit(source: str, replacements) -> str:
    """Apply ``(old, new[, count])`` string replacements, each expected ``count`` (default 1) times."""
    for old, new, *count in replacements:
        if source.count(old) != (count[0] if count else 1):
            raise SystemExit(f"unexpected occurrences of {old!r} in the generator")
        source = source.replace(old, new)
    return source


def run_fig2(source: str, out: Path, paper_style: bool = False) -> Path:
    namespace = {"__file__": str(GENERATOR), "__name__": "make_real_figures"}
    exec(compile(source, str(GENERATOR), "exec"), namespace)
    namespace["FIG"] = out
    if paper_style:
        plt = C.setup_matplotlib()
        # the generator's light-grey surface becomes white; its ink, grid and spine settings are kept
        namespace["SURFACE"] = "white"
        plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                             "savefig.facecolor": "white", "hatch.color": "white"})
    namespace["fig2"]()
    return out / GENERATED


def pixels(path: Path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=int)


def main():
    source = GENERATOR.read_text()
    relabelled = edit(source, RELABEL)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "check").mkdir()
        check = run_fig2(relabelled, tmp / "check").with_suffix(".png")
        same = hashlib.md5(check.read_bytes()).digest() == hashlib.md5(REFERENCE.read_bytes()).digest()
        if not same:
            a, b = pixels(check), pixels(REFERENCE)
            if a.shape != b.shape or np.abs(a - b).max() > 0:
                raise SystemExit("the generator no longer reproduces the L2A_011 figure; not restyling")
        print(f"relabelled generator reproduces {REFERENCE.name}: {'byte-identical' if same else 'pixel-identical'}")
    OUT.mkdir(parents=True, exist_ok=True)
    out = run_fig2(edit(relabelled, STYLE), OUT, paper_style=True)
    for ext in (".pdf", ".png"):
        target = OUT / f"{NAME}{ext}"
        out.with_suffix(ext).replace(target)
        print("wrote", target)


if __name__ == "__main__":
    main()
