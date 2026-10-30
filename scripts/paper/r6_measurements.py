"""A2L_005 §1 (R-6): filter bypass and what the residual carries, deployed r7-taufree checkpoints.

Writes ``paper_r6_bypass.csv`` and ``paper_r6_residual_share.csv`` to
``REFACTOR_SPECS/final_results/data/``.  Stops before writing anything if the
gate numbers (KALNN two-mass test RMSE iiwa 3.402, UR10 4.677) or a file's
SHA-256 do not reproduce.

    ~/Projects/dynamic_model_nn/.venv/bin/python scripts/paper/r6_measurements.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import common as C

GATE = {"iiwa_drive": 3.402, "ur10": 4.677}
# final_results/12 §5.1: KALNN two-mass, gain-shift file, filter on -> bypassed
GAINSHIFT_REPORTED = {"iiwa_drive": (3.19, 3.30), "ur10": (4.40, 5.83)}
TANH_EPS = 1e-3


def link_friction(audit: pd.DataFrame, variant: str, dq_col: str) -> np.ndarray:
    """``c tanh(dq_link / 1e-3) + v dq_link`` per row and joint, from the per-row r7 friction law."""
    cols = []
    for j, name in enumerate(C.JOINT_NAMES[variant]):
        c = audit[f"link_coulomb__{name}"].to_numpy(float)
        v = audit[f"link_viscous__{name}"].to_numpy(float)
        w = audit[f"{dq_col}{j}"].to_numpy(float)
        cols.append(c * np.tanh(w / TANH_EPS) + v * w)
    return np.column_stack(cols)


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main():
    for key, path in C.FILES.items():
        digest = C.sha256(path)
        if digest != C.SHA256[key]:
            raise SystemExit(f"SHA-256 mismatch {key}: {digest}")
    deployed = C.deployed_checkpoints()
    print(deployed.to_string())

    bypass_rows, share_rows, notes = [], [], {}
    for variant in C.VARIANTS:
        labels = C.JOINT_LABELS[variant]
        head = C.BURN_IN[variant]
        dataset_cache = {}
        audit = None
        for file in ("test", "gainshift"):
            path = C.FILES[(variant, file)]
            for _, r in deployed[deployed.variant == variant].iterrows():
                two_mass = r.model.endswith("twomass")
                if file == "gainshift" and not two_mass:
                    continue
                ck = C.DMN / r.checkpoint
                normalization = torch.load(ck, map_location="cpu", weights_only=False)["normalization"]
                norm_key = (file, json.dumps(normalization, sort_keys=True))
                if norm_key not in dataset_cache:
                    dataset_cache[norm_key] = C.load_dataset(ck, path)[0]
                o = C.rollout(ck, path, "test", head, dataset=dataset_cache[norm_key])
                y = C.scored(o, "target")
                e_on = C.scored(o, "pred") - y
                e_by = C.scored(o, "bypass") - y
                rmse_on = C.rmse(e_on)
                print(f"{variant:10s} {file:9s} {r.model:14s} s{r.seed} rmse_on {rmse_on:.6f} "
                      f"(reported test {r.test_rmse:.6f}) bypassed {C.rmse(e_by):.6f}  n={len(y)}", flush=True)
                if file == "test" and abs(rmse_on - r.test_rmse) > 1e-3:
                    raise SystemExit(f"STOP: {variant} {r.model} test RMSE {rmse_on:.6f} != {r.test_rmse:.6f}")
                if file == "test" and r.model == "kalnn_twomass" and abs(rmse_on - GATE[variant]) > 1e-3:
                    raise SystemExit(f"STOP: gate {variant} {rmse_on:.6f} != {GATE[variant]}")
                if two_mass:
                    for j, label in [(None, "all")] + list(enumerate(labels)):
                        a = e_on if j is None else e_on[:, j]
                        b = e_by if j is None else e_by[:, j]
                        bypass_rows.append(dict(variant=variant, **{"class": r.model}, checkpoint=r.checkpoint,
                                                file=file, joint=label, rmse_on=C.rmse(a), rmse_bypassed=C.rmse(b),
                                                ratio=C.rmse(b) / C.rmse(a)))
                    if file == "gainshift" and r.model == "kalnn_twomass":
                        notes[f"{variant}_gainshift_reported"] = GAINSHIFT_REPORTED[variant]
                        notes[f"{variant}_gainshift_mine"] = (round(rmse_on, 4), round(C.rmse(e_by), 4))
                if file != "test":
                    continue
                # (b) what the residual carries
                if audit is None:
                    cols = ["bag", "t"] + [f"dq_link{j}" for j in range(len(labels))] \
                        + [f"dq_link_clean{j}" for j in range(len(labels))] \
                        + [f"link_{k}__{n}" for n in C.JOINT_NAMES[variant] for k in ("coulomb", "viscous")]
                    audit = pd.read_parquet(path, columns=cols)
                    assert len(audit) == len(o["t"]) and np.allclose(audit["t"].to_numpy(), o["t"])
                    fl_all = link_friction(audit, variant, "dq_link")
                    fl_clean_all = link_friction(audit, variant, "dq_link_clean")
                    notes[f"{variant}_link_friction_law"] = {
                        n: sorted({(round(c, 6), round(v, 6)) for c, v in zip(
                            audit[f"link_coulomb__{n}"].to_numpy()[o["rows"]],
                            audit[f"link_viscous__{n}"].to_numpy()[o["rows"]])})
                        for n in C.JOINT_NAMES[variant]}
                res = C.scored(o, "res")
                fl = fl_all[o["rows"]][o["keep"]]
                fl_clean = fl_clean_all[o["rows"]][o["keep"]]
                for j, label in [(None, "pooled")] + list(enumerate(labels)):
                    rr = res if j is None else res[:, j]
                    yy = y if j is None else y[:, j]
                    ff = fl if j is None else fl[:, j]
                    fc = fl_clean if j is None else fl_clean[:, j]
                    share_rows.append(dict(variant=variant, **{"class": r.model}, joint=label,
                                           rms_residual=C.rmse(rr), rms_target=C.rmse(yy),
                                           share=C.rmse(rr) / C.rmse(yy),
                                           corr_residual_vs_link_friction=corr(rr, ff),
                                           n_rows=int(len(yy)),
                                           _corr_clean=corr(rr, fc)))

    bypass = pd.DataFrame(bypass_rows)
    share = pd.DataFrame(share_rows)
    bypass.to_csv(C.DATA_OUT / "paper_r6_bypass.csv", index=False)
    share.drop(columns="_corr_clean").to_csv(C.DATA_OUT / "paper_r6_residual_share.csv", index=False)
    pd.set_option("display.width", 200)
    print(bypass.drop(columns="checkpoint").to_string(float_format="%.4f"))
    print(share.to_string(float_format="%.4f"))
    print(json.dumps(notes, indent=1, default=str))


if __name__ == "__main__":
    main()
