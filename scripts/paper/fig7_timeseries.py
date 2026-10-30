"""A2L_005 §3 (R-1): Fig. 7, time series on a real iiwa bag and a simulated UR10 test bag.

(a) real iiwa, file B (``iiwa_fast_all_01``), the bag whose KALNN two-mass
    (deployed) RMSE is closest to the file's median over its 10 bags
    (``real_zero_shot_per_bag.csv``, raw input, no correction); joints A2 and A1.
    Lines: measured tau_s, rigid URDF (RNEA, ``real_eval/rnea.py``), KALNN
    residual, KALNN two-mass (deployed seeds, filter step 1 ms).
(b) simulated UR10 test file, the bag whose seed-pooled KALNN two-mass /
    KALNN residual RMSE ratio (``per_bag_errors.csv``) is closest to 0.97;
    joint lift: tau_s and the two predictions; the true deflection ``defl``
    and the latent delta_hat in mrad.

Only scored rows (after the burn-in) are plotted.  The 4 s window is chosen
by one rule, the same for both rows: the window (0.5 s grid, inside the
scored part) with the largest RMS motor velocity of the first plotted joint.

    ~/Projects/dynamic_model_nn/.venv/bin/python scripts/paper/fig7_timeseries.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common as C

REAL_B = C.ERS / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01/dataset/kuka_lbr_iiwa_14_r820.parquet"
IIWA_URDF = C.ERS / "assets/robots/kuka_lbr_iiwa_14_r820/description/kuka_lbr_iiwa_14_r820.urdf"
WINDOW = 4.0
UR10_TARGET_RATIO = 0.97


def choose_real_bag():
    d = pd.read_csv(C.DATA_OUT / "real_zero_shot_per_bag.csv")
    dep = C.deployed_checkpoints()
    seed = int(dep[(dep.variant == "iiwa_drive") & (dep.model == "kalnn_twomass")].seed.iloc[0])
    k = d[(d.robot == "iiwa_drive") & (d.file == "B") & (d.input == "raw") & (d.calib == "none")
          & (d.model == "kalnn_twomass") & (d.seed == seed)].copy()
    med = float(k.rmse.median())
    k["dist"] = (k.rmse - med).abs()
    # Ties (an even count puts the median midway between two bags): lowest bag index.
    k["index"] = k.bag.str.extract(r"(\d+)$")[0].astype(int)
    k = k.sort_values(["dist", "index"])
    tie = bool(len(k) > 1 and abs(k.dist.iloc[0] - k.dist.iloc[1]) < 1e-9)
    return k.bag.iloc[0], med, float(k.rmse.iloc[0]), tie, k.bag.iloc[1] if tie else None


def choose_ur10_bag():
    p = pd.read_csv(C.DATA_OUT / "per_bag_errors.csv")
    g = p[(p.variant == "ur10") & p.model.isin(["kalnn_twomass", "kalnn_tau_res"])].groupby(["model", "bag"])[
        ["sse", "n"]].sum()
    rm = lambda m: np.sqrt(g.loc[m].sse / g.loc[m].n)  # noqa: E731
    ratio = rm("kalnn_twomass") / rm("kalnn_tau_res")
    bag = (ratio - UR10_TARGET_RATIO).abs().idxmin()
    G = g.groupby("model").sum()
    pooled = float(np.sqrt(G.loc["kalnn_twomass"].sse / G.loc["kalnn_twomass"].n)
                   / np.sqrt(G.loc["kalnn_tau_res"].sse / G.loc["kalnn_tau_res"].n))
    return bag, float(ratio[bag]), pooled


def window(t, speed, length=WINDOW, step=0.5):
    """Start of the ``length`` window (``step`` grid) with the largest RMS ``speed``, inside ``t``."""
    best, start = -1.0, t[0]
    s = t[0]
    while s + length <= t[-1] + 1e-9:
        m = (t >= s) & (t < s + length)
        v = float(np.sqrt(np.mean(speed[m] ** 2)))
        if v > best:
            best, start = v, s
        s += step
    return float(start)


def real_panel():
    sys.path.insert(0, str(C.REAL_EVAL))
    import data as rdata
    import rnea
    bag, med, bag_rmse, tie, other = choose_real_bag()
    dep = C.deployed_checkpoints()
    dep = dep[dep.variant == "iiwa_drive"].set_index("model")
    d = rdata.load(REAL_B)
    idx = d["bag_index"][bag]
    t = d["t"][idx] - d["t"][idx][0]
    keep = t >= C.BURN_IN["iiwa_drive"]
    robot = rnea.Robot(str(IIWA_URDF), C.JOINT_NAMES["iiwa_drive"])
    rig = robot.rnea(d["q"][idx], d["dq"][idx], d["ddq"][idx])
    lines = {"rigid URDF": rig}
    meta_models = {}
    dataset = None
    for name, model in [("KALNN residual", "kalnn_tau_res"), ("KALNN two-mass", "kalnn_twomass")]:
        ck = C.DMN / dep.loc[model, "checkpoint"]
        dataset, _ = C.load_dataset(ck, REAL_B)
        o = C.rollout(ck, REAL_B, "all", 0.0, dataset=dataset)
        assert np.allclose(o["target"][idx], d["ft"][idx], atol=1e-4)
        lines[name] = o["pred"][idx]
        step = getattr(o["model"], "time_step", None)
        meta_models[name] = dict(model=model, seed=int(dep.loc[model, "seed"]), checkpoint=dep.loc[model, "checkpoint"],
                                 filter_time_step=step)
        if step is not None:
            assert abs(step - 1e-3) < 1e-12, step
    y = d["ft"][idx]
    # Cross-check against real_zero_shot_per_bag.csv (all joints, scored rows of the bag).
    ref = pd.read_csv(C.DATA_OUT / "real_zero_shot_per_bag.csv")
    ref = ref[(ref.robot == "iiwa_drive") & (ref.file == "B") & (ref.input == "raw") & (ref.calib == "none")
              & (ref.bag == bag)]
    checks = {}
    for name, key, seed in [("rigid URDF", "rigid_URDF", -1),
                            ("KALNN residual", "kalnn_tau_res", int(dep.loc["kalnn_tau_res", "seed"])),
                            ("KALNN two-mass", "kalnn_twomass", int(dep.loc["kalnn_twomass", "seed"]))]:
        mine = float(C.rmse((lines[name] - y)[keep]))
        theirs = float(ref[(ref.model == key) & (ref.seed == seed)].rmse.iloc[0])
        checks[name] = dict(mine=mine, reported=theirs)
        print(f"real B {bag} {name:15s} all-joint RMSE mine {mine:.4f} reported {theirs:.4f}")
        if abs(mine - theirs) > 1e-3:
            raise SystemExit(f"real per-bag RMSE mismatch for {name}")
    j2 = C.JOINT_LABELS["iiwa_drive"].index("A2")
    ts, tk = t[keep], keep
    start = window(ts, d["dq"][idx][keep][:, j2])
    return dict(bag=bag, median=med, bag_rmse=bag_rmse, tie=tie, tie_with=other, t=t, keep=keep, y=y, lines=lines,
                start=start, models=meta_models, checks=checks, tk=tk)


def sim_panel():
    bag, ratio, pooled = choose_ur10_bag()
    dep = C.deployed_checkpoints()
    dep = dep[dep.variant == "ur10"].set_index("model")
    path = C.FILES[("ur10", "test")]
    head = C.BURN_IN["ur10"]
    out, models = {}, {}
    dataset = None
    for name, model in [("KALNN residual", "kalnn_tau_res"), ("KALNN two-mass", "kalnn_twomass")]:
        ck = C.DMN / dep.loc[model, "checkpoint"]
        if dataset is None:
            dataset, _ = C.load_dataset(ck, path)
        o = C.rollout(ck, path, "test", head, dataset=dataset)
        out[name] = o
        models[name] = dict(model=model, seed=int(dep.loc[model, "seed"]), checkpoint=dep.loc[model, "checkpoint"])
    o = out["KALNN two-mass"]
    idx = np.flatnonzero(o["bag"] == bag)
    t = o["t"][idx] - o["t"][idx][0]
    keep = t >= head
    j = C.JOINT_LABELS["ur10"].index("lift")
    audit = pd.read_parquet(path, columns=["t", "bag", f"defl{j}", f"dq_motor_clean{j}"])
    assert np.allclose(audit["t"].to_numpy()[idx], o["t"][idx])
    defl = audit[f"defl{j}"].to_numpy(float)[idx]
    speed = audit[f"dq_motor_clean{j}"].to_numpy(float)[idx]
    y = o["target"][idx]
    delta_hat = o["delta"][idx][:, j]
    corr = float(np.corrcoef(delta_hat[keep], defl[keep])[0, 1])
    start = window(t[keep], speed[keep])
    lines = {name: out[name]["pred"][idx] for name in out}
    return dict(bag=bag, ratio=ratio, pooled=pooled, t=t, keep=keep, y=y, lines=lines, defl=defl,
                delta_hat=delta_hat, corr=corr, start=start, models=models, joint=j)


def main():
    plt = C.setup_matplotlib()
    a = real_panel()
    b = sim_panel()
    colour = {"rigid URDF": (C.C_RIGID, "--"), "KALNN residual": (C.C_RES, "-"), "KALNN two-mass": (C.C_2M, "-")}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.2))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.10, wspace=0.22, hspace=0.75)
    # A2L_013: the proposed model is shown as "2M-KALNN" (display labels only; data keys unchanged).
    label = {"rigid URDF": "rigid URDF", "KALNN residual": "KALNN residual", "KALNN two-mass": "2M-KALNN"}
    short = {"rigid URDF": "URDF", "KALNN residual": "KALNN residual", "KALNN two-mass": "2M-KALNN"}
    meta = {"real_iiwa_B": {}, "sim_ur10": {}}

    # (a) real iiwa, A2 and A1
    w = (a["t"] >= a["start"]) & (a["t"] < a["start"] + WINDOW) & a["keep"]
    meta["real_iiwa_B"] = dict(file="iiwa_fast_all_01", bag=a["bag"], kalnn_twomass_bag_rmse=a["bag_rmse"],
                               file_median=a["median"], median_tie_with=a["tie_with"],
                               window_s=[a["start"], a["start"] + WINDOW], models=a["models"],
                               all_joint_rmse_check=a["checks"], per_joint_rmse={})
    for ax, joint in zip(axes[0], ["A2", "A1"]):
        j = C.JOINT_LABELS["iiwa_drive"].index(joint)
        ax.plot(a["t"][w], a["y"][w, j], color=C.C_MEAS, lw=0.8, label=r"measured / true $\tau_s$", zorder=4)
        rm = {}
        for name, (c, ls) in colour.items():
            rm[name] = float(C.rmse((a["lines"][name][:, j] - a["y"][:, j])[a["keep"]]))
            ax.plot(a["t"][w], a["lines"][name][w, j], color=c, ls=ls, lw=0.8, label=label[name])
        meta["real_iiwa_B"]["per_joint_rmse"][joint] = rm
        ax.set_title(f"(a) real iiwa, file B, {a['bag'].replace('_', ' ')}: {joint}\nRMSE [N·m]: "
                     + ", ".join(f"{short[k]} {v:.2f}" for k, v in rm.items()), fontsize=8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{joint} torque [N·m]")
        ax.margins(x=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))

    # (b) simulated UR10, lift
    j = b["joint"]
    w = (b["t"] >= b["start"]) & (b["t"] < b["start"] + WINDOW) & b["keep"]
    ax = axes[1, 0]
    ax.plot(b["t"][w], b["y"][w, j], color=C.C_MEAS, lw=0.8, zorder=4)
    rm = {}
    for name in ("KALNN residual", "KALNN two-mass"):
        c, ls = colour[name]
        rm[name] = float(C.rmse((b["lines"][name][:, j] - b["y"][:, j])[b["keep"]]))
        ax.plot(b["t"][w], b["lines"][name][w, j], color=c, ls=ls, lw=0.8)
    ax.set_title(f"(b) simulated UR10, test bag {b['bag'].replace('_f0_mujoco', '')}: lift\nRMSE [N·m]: "
                 + ", ".join(f"{short[k]} {v:.2f}" for k, v in rm.items()), fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lift torque [N·m]")
    ax.margins(x=0)
    ax = axes[1, 1]
    ax.plot(b["t"][w], 1e3 * b["defl"][w], color=C.C_MEAS, lw=0.9, label=r"true $\theta - q$")
    ax.plot(b["t"][w], 1e3 * b["delta_hat"][w], color=C.C_2M, lw=0.9,
            label=r"2M-KALNN $\hat\delta$")
    ax.set_title("(b) lift deflection, same bag\n" + rf"corr($\hat\delta$, $\theta - q$) over the bag {b['corr']:.2f}",
                 fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deflection [mrad]")
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
    ax.margins(x=0)
    meta["sim_ur10"] = dict(file=str(C.FILES[("ur10", "test")].name), bag=b["bag"],
                            bag_ratio_seed_pooled=b["ratio"], pooled_ratio_recomputed=b["pooled"],
                            window_s=[b["start"], b["start"] + WINDOW], models=b["models"],
                            per_joint_rmse={"lift": rm}, corr_delta_hat_vs_defl_lift_bag=b["corr"])
    meta["window_rule"] = ("4 s window on a 0.5 s grid inside the scored rows, largest RMS motor velocity of the "
                           "first plotted joint (A2 real, lift sim); RMSEs in legends over all scored rows of the bag")
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"fig_timeseries.{ext}", dpi=300)
    (C.DATA_OUT / "paper_r1_timeseries_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
