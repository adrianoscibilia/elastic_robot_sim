"""The percentile-loss variants of the fair baselines (`03_fair_comparison_plan.md` F6, F8).

The LNN two-mass runs trained with ``percentile_mse@90`` (``model_workflow.custom_loss``:
the 90th percentile over a batch's samples of the per-sample mean of the
scaled squared error ``(s_j (tau_hat_j - tau_j))^2``), so the fair baselines
are also fitted that way, with the LNNs' optimiser and schedule:

* Adam at the LNN's learning rate, constant; batches as the LNNs' (the
  residual class: ``batch_size`` shuffled train rows; the two-mass class:
  ``model_workflow.sequence_batch_windows`` windows, one on these platforms);
* early stopping on the validation RMSE in physical units (relative
  ``min_delta`` 1e-4, patience, at most ``epochs``), best weights restored;
* one run per seed (the seed drives the batch order), the best-validation
  seed deployed.

**ID-rigid-fair, P90**: the base vector of ID-rigid-fair's grid point,
normalised by its columns' RMS (``beta_k = u_k / c_k``) so that Adam's
step means the same torque change for every parameter; started at the
least-squares solution.

**ID-2M-fair, P90**: ``[theta (log-Cholesky), F, kappa, eta]`` as the
least-squares ID-2M-fair, started at its stage 1 (the consistent rigid
fit) with K, zeta at the F4 init; stage 2 trains ``kappa, eta`` only,
stage 3 everything.  The filter is ``dynamic_model_nn/twomass_filter.py``
(its prefix scan, the LNNs' own code); ``M_hat(theta)`` is the mass
regressor contracted with ``pi(theta)``, so gradients through ``M_hat`` and
``M_bar`` are exact.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .fair import (
    BagRows, FairLayout, FairModel, FairPriors, LinkDynamics, consistent_inertials, head_mask, sequence_windows,
)


@dataclass(frozen=True)
class AdamSettings:
    learning_rate: float
    batch_size: int
    epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4            # model_workflow.VAL_MIN_DELTA
    loss_percentile: float = 90.0
    device: str = "auto"


def percentile_loss(torch, prediction, target, scale, percentile: float):
    """``custom_loss`` with ``loss_type="percentile_mse"``, ``channel_scale`` and no residual (no L1)."""
    per_sample = (((prediction - target) * scale) ** 2).mean(dim=-1)
    return torch.quantile(per_sample.reshape(-1), percentile / 100.0)


def _device(torch, name: str):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _train(torch, params, step_loss: Callable, batches: Callable, val_rmse: Callable, settings: AdamSettings,
           seed: int, log: Callable[[str], None] | None, label: str) -> dict[str, Any]:
    """The LNNs' loop: Adam, val RMSE each epoch, patience, best weights restored."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(params, lr=settings.learning_rate)
    best = {"val": val_rmse(), "epoch": -1, "state": [p.detach().clone() for p in params]}
    history, patience, started = [{"epoch": -1, "val_rmse": best["val"]}], 0, time.perf_counter()
    for epoch in range(settings.epochs):
        total, count = 0.0, 0
        for batch in batches(generator):
            optimizer.zero_grad(set_to_none=True)
            loss = step_loss(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        val = val_rmse()
        history.append({"epoch": epoch, "loss": total / max(count, 1), "val_rmse": val})
        if math.isfinite(val) and val < best["val"] * (1.0 - settings.min_delta):
            best, patience = {"val": val, "epoch": epoch, "state": [p.detach().clone() for p in params]}, 0
        else:
            patience += 1
        if log and (epoch % 10 == 0 or patience == 0):
            log(f"  {label} seed {seed} epoch {epoch:3d}: loss {total / max(count, 1):.5g} val {val:.4f} "
                f"(best {best['val']:.4f} @ {best['epoch']}) [{time.perf_counter() - started:.0f} s]")
        if patience >= settings.patience:
            break
    with torch.no_grad():
        for p, v in zip(params, best["state"]):
            p.copy_(v)
    return {"val_rmse": best["val"], "best_epoch": best["epoch"], "epochs_run": len(history) - 1,
            "history": history}


# ------------------------------------------------------------------------------------------- ID-rigid-fair, P90


def fit_rigid_p90(rigid, train: Sequence[BagRows], val: Sequence[BagRows], scale: np.ndarray,
                  settings: AdamSettings, seed: int, log: Callable[[str], None] | None = print
                  ) -> tuple[FairModel, dict[str, Any]]:
    """``rigid`` is the least-squares :class:`~.fair.RigidFit` of the same ``eps`` (its base and start)."""
    import torch

    device = _device(torch, settings.device)
    layout = FairLayout(rigid.model.n)
    cols = rigid.base.independent
    as_t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)  # noqa: E731
    W = as_t(np.concatenate([r.rows(layout, rigid.model.epsilon)[:, :, cols] for r in train]))   # [N, n, r]
    Y = as_t(np.concatenate([r.bag.tau_J for r in train]))
    Wv = as_t(np.concatenate([r.rows(layout, rigid.model.epsilon)[:, :, cols] for r in val]))
    Yv = as_t(np.concatenate([r.bag.tau_J for r in val]))
    c = torch.sqrt((W.double() ** 2).mean(dim=(0, 1))).clamp_min(1e-12).to(device)              # column RMS
    u = torch.nn.Parameter((torch.as_tensor(rigid.beta, device=device) * c).float())
    s = as_t(scale)
    c32 = c.float()

    def predict(Wx):
        return Wx @ (u / c32)

    def val_rmse():
        with torch.no_grad():
            return float(torch.sqrt(((predict(Wv) - Yv).double() ** 2).mean()))

    def batches(generator):
        order = torch.randperm(len(W), generator=generator).to(device)
        for lo in range(0, len(W), settings.batch_size):
            yield order[lo:lo + settings.batch_size]

    def step_loss(index):
        return percentile_loss(torch, predict(W[index]), Y[index], s, settings.loss_percentile)

    record = _train(torch, [u], step_loss, batches, val_rmse, settings, seed, log, "ID-rigid-fair P90")
    beta = (u.detach().double() / c).cpu().numpy()
    full = rigid.model.full.copy()          # dependent and zero columns at the URDF, as ID-rigid-fair's
    full[rigid.base.independent] = beta - (rigid.base.relation @ full[rigid.base.dependent]
                                           if len(rigid.base.dependent) else 0.0)
    model = FairModel("id_rigid_fair_p90", layout.n, full, rigid.model.epsilon, info={
        "seed": seed, "val_rmse": record["val_rmse"], "best_epoch": record["best_epoch"],
        "epochs_run": record["epochs_run"], "loss": f"percentile_mse@{settings.loss_percentile:g}",
        "optimiser": f"Adam lr {settings.learning_rate:g}, batch {settings.batch_size} rows, from the LS solution"})
    return model, record


# ---------------------------------------------------------------------------------------------- ID-2M-fair, P90


def _log_cholesky_torch(torch, theta):
    """:func:`~.fair.inertial_from_log_cholesky` for ``[n, 10]`` blocks, differentiable."""
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = theta.unbind(-1)
    zero, one = torch.zeros_like(alpha), torch.ones_like(alpha)
    U = torch.exp(alpha)[:, None, None] * torch.stack([
        torch.stack([torch.exp(d1), s12, s13, t1], -1), torch.stack([zero, torch.exp(d2), s23, t2], -1),
        torch.stack([zero, zero, torch.exp(d3), t3], -1), torch.stack([zero, zero, zero, one], -1)], -2)
    J = U @ U.transpose(-1, -2)
    sigma = J[:, :3, :3]
    inertia = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1)[:, None, None] * torch.eye(3, dtype=J.dtype,
                                                                                         device=J.device) - sigma
    return torch.stack([J[:, 3, 3], J[:, 0, 3], J[:, 1, 3], J[:, 2, 3], inertia[:, 0, 0], inertia[:, 0, 1],
                        inertia[:, 1, 1], inertia[:, 0, 2], inertia[:, 1, 2], inertia[:, 2, 2]], -1).reshape(-1)


def lnn_filter(repo: str | Path):
    """``two_mass_filter`` of ``dynamic_model_nn/twomass_filter.py`` (the LNNs' own filter)."""
    path = str(Path(repo).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from twomass_filter import two_mass_filter

    return two_mass_filter


def fit_two_mass_p90(stage1: FairModel, train: Sequence[BagRows], val: Sequence[BagRows], rigid_model: tuple,
                     priors: FairPriors, scale: np.ndarray, mass_regressors: list[np.ndarray], settings: AdamSettings,
                     seed: int, lnn_repo: str | Path, log: Callable[[str], None] | None = print
                     ) -> tuple[FairModel, dict[str, Any]]:
    """Stages 2 and 3 of plan Sec 3.2 with the P90 loss and Adam, from ``stage1`` (consistent rigid fit)."""
    import torch

    two_mass_filter = lnn_filter(lnn_repo)
    device = _device(torch, settings.device)
    n = stage1.n
    layout = FairLayout(n)
    inertial_at = torch.as_tensor(layout.inertial_slice(), device=device)
    friction_at = np.r_[layout.friction_slices()[0], layout.friction_slices()[1]]
    as_t = lambda x, dt=torch.float64: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)  # noqa
    windows, U, targets, Mreg = [], [], [], []
    loss_from = None
    for b, rows in enumerate(train):
        W = rows.rows(layout, stage1.epsilon).astype(np.float32)
        for start, stop in sequence_windows(rows.bag.t, priors):
            keep = head_mask(rows.bag.t[start:stop], priors.burn_in_s)
            loss_from = int(np.argmax(keep))
            windows.append((b, start, stop))
            U.append(W[start:stop])
            targets.append(rows.bag.tau_J[start:stop][loss_from:])
    U = as_t(np.stack(U), torch.float32)                                     # [B, L, n, 12 n]
    targets = as_t(np.stack(targets))
    mass_gpu = [as_t(m, torch.float32) for m in mass_regressors]             # [T, n, n, 10 n] per bag
    val_keep = int(np.argmax(head_mask(val[0].bag.t, priors.burn_in_s)))
    U_val = [as_t(r.rows(layout, stage1.epsilon), torch.float64) for r in val]
    target_val = as_t(np.stack([r.bag.tau_J[val_keep:] for r in val]))
    s = as_t(scale)
    k_low, k_high = as_t(priors.k_low), as_t(priors.k_high)
    z_low, z_high = as_t(priors.zeta_low), as_t(priors.zeta_high)
    theta = torch.nn.Parameter(as_t(np.asarray(stage1.info["log_cholesky"]).reshape(n, 10)))
    friction = torch.nn.Parameter(as_t(stage1.full[friction_at]))
    kappa0, eta0 = priors.logits(priors.k_init, priors.zeta_init)
    kappa, eta = torch.nn.Parameter(as_t(kappa0)), torch.nn.Parameter(as_t(eta0))

    def full():
        pi = _log_cholesky_torch(torch, theta)
        out = torch.zeros(12 * n, dtype=torch.float64, device=device)
        out = out.index_put((inertial_at,), pi)
        return out.index_put((torch.as_tensor(friction_at, device=device),), friction), pi

    def transmission():
        return k_low + (k_high - k_low) * torch.sigmoid(kappa), z_low + (z_high - z_low) * torch.sigmoid(eta)

    def predict_window(index: int):
        b, start, stop = windows[index]
        vector, pi = full()
        tau_r = U[index].double() @ vector
        M = torch.einsum("lijc,c->lji", mass_gpu[b][start:stop].double(), pi)   # M[:, j, i] = (M e_i)_j
        M = 0.5 * (M + M.transpose(-1, -2))
        K, zeta = transmission()
        D = 2.0 * zeta * torch.sqrt(K * torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=0))
        tau_s, _, _ = two_mass_filter(tau_r[None], M[None], K, D, priors.step, coupling="full")
        return tau_s[0, loss_from:]

    def val_rmse():
        with torch.no_grad():
            vector, pi = full()
            dyn = LinkDynamics(rigid_model, pi.cpu().numpy(), consistent=True)
            K, zeta = transmission()
            errors = []
            for r, Uv, target in zip(val, U_val, target_val):
                M = as_t(dyn.inertia(r.theta))
                D = 2.0 * zeta * torch.sqrt(K * torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=0))
                tau_s, _, _ = two_mass_filter((Uv @ vector)[None], M[None], K, D, priors.step, coupling="full")
                errors.append(tau_s[0, val_keep:] - target)
            value = float(torch.sqrt((torch.stack(errors) ** 2).mean()))
            return value if math.isfinite(value) else float("inf")

    def batches(generator):
        for index in torch.randperm(len(windows), generator=generator).tolist():
            yield index

    def step_loss(index):
        return percentile_loss(torch, predict_window(index), targets[index], s, settings.loss_percentile)

    torch.manual_seed(seed)
    stage2 = _train(torch, [kappa, eta], step_loss, batches, val_rmse, settings, seed, log, "ID-2M-fair P90 stage 2")
    stage3 = _train(torch, [theta, friction, kappa, eta], step_loss, batches, val_rmse, settings, seed, log,
                    "ID-2M-fair P90 stage 3")
    with torch.no_grad():
        vector, _ = full()
        K, zeta = transmission()
    model = FairModel("id_2m_fair_p90", n, vector.cpu().numpy(), stage1.epsilon, K.cpu().numpy(),
                      zeta.cpu().numpy(), info={
                          "log_cholesky": theta.detach().cpu().numpy().reshape(-1).tolist(), "seed": seed,
                          "val_rmse": stage3["val_rmse"], "stage2": {k: v for k, v in stage2.items() if k != "history"},
                          "stage3": {k: v for k, v in stage3.items() if k != "history"},
                          "loss": f"percentile_mse@{settings.loss_percentile:g}",
                          "optimiser": f"Adam lr {settings.learning_rate:g}, one window per batch, "
                                       "dynamic_model_nn twomass_filter (scan), from ID-2M-fair's stage 1"})
    return model, {"stage2": stage2, "stage3": stage3}
