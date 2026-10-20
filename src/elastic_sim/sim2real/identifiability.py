"""What the data can see (S2R_01 T-4.3, S2R_04 R-2/A-11).

Input: the normalised data Jacobian ``J`` of the stacked least-squares
residual (rows weighted exactly as :class:`~.loss.Loss` weighs them) with
respect to ``u = f(theta) / sigma``, i.e. one column per parameter in units
of its prior sigma, and the prior weight ``lambda`` of the same objective
(the prior rows are ``sqrt(lambda) * u``). Then:

* **sensitivity** ``s_i = |J_i|``: residual change per prior sigma; relative
  to the largest;
* **diagonal data share** ``d_i = s_i^2 / (s_i^2 + lambda)``: S2R_04 A-11's
  ``diag(J^T J) / (diag(J^T J) + sigma^-2)`` in the objective's own
  normalisation. A diagnostic column only (S2R_06 F-2): it sees one column
  at a time, so two collinear parameters can both look data-determined
  while only their combination is;
* **data share** (A-11 as amended by S2R_06 F-2) ``d_post_i = 1 - lambda *
  [(J^T J + lambda I)^-1]_ii``: the fraction of the prior variance the data
  removes, *with* the correlations (the posterior sd in prior sigmas is
  ``sqrt(1 - d_post)``); ``d_post < 0.9`` = **prior-determined**. Analysed
  parameters that are not free (K) have no posterior: they are flagged by
  ``d < 0.9``, which suffices since ``d >= d_post``;
* **SVD** of the column-normalised ``J``: singular values, condition number,
  the **collinearity index** ``gamma = 1 / s_min`` (Brun et al. 2001) of the
  set and per parameter ``gamma_i = sqrt([(J~^T J~)^+]_ii)``;
* **collinear pairs**: |cos| between columns, the top ones and the
  sensor-gain / link-mass pairs (S2R_04 R-3);
* **recommended frozen set**: sensitivity < 1e-3 of the largest, then
  greedily the parameter that dominates the weakest singular direction until
  ``gamma <= 30`` (S2R_01 T-4.3). Reported; the procedure that acts is
  A-12's (:func:`freeze_plan`).

:func:`labels` is S2R_08 A-12' (the default): ``calibrated`` iff d_post >=
0.9 on the full free set, once; the not-calibrated ones list their
**partners**, the free parameters with |posterior correlation| >= 0.5.

:func:`freeze_plan` is S2R_06 A-12's freeze step (``mode: freeze_refit``): drop the free parameter
with the lowest d_post, recompute d_post on the rest (the Jacobian at the
point with the dropped one at its freeze value), repeat until every free
parameter has ``d_post >= d_post_min``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

SENSITIVITY_REL_MIN = 1e-3
COLLINEARITY_MAX = 30.0
#: S2R_06 F-2: the data share is d_post, data-determined from 0.9.
PRIOR_DETERMINED_BELOW = 0.9
TOP_PAIRS = 15
#: S2R_08 A-12': a not-calibrated parameter's partners are the free
#: parameters with |posterior correlation| >= this.
PARTNER_MIN = 0.5


def analyse(jacobian_u: np.ndarray, names: Sequence[str], *, prior_weight: float,
            free: Sequence[str] | None = None, sensitivity_rel_min: float = SENSITIVITY_REL_MIN,
            collinearity_max: float = COLLINEARITY_MAX, partner_min: float = PARTNER_MIN) -> dict[str, Any]:
    """The identifiability block. ``free`` (default: all ``names``) is the
    optimised set; other columns (frozen parameters analysed for the record,
    e.g. K) get sensitivity, d and pairs, not the posterior or freeze rule."""
    J = np.asarray(jacobian_u, dtype=float)
    names = list(names)
    free = list(names if free is None else free)
    lam = float(prior_weight)
    s = np.linalg.norm(J, axis=0)
    s_max = float(s.max()) if s.size and s.max() > 0 else 1.0
    d = s ** 2 / (s ** 2 + lam)
    columns = np.where(s > 0, s, 1.0)
    Jn = J / columns
    gram = Jn.T @ Jn
    cos = np.clip(gram, -1.0, 1.0)

    free_idx = [names.index(n) for n in free]
    Jf = J[:, free_idx]
    shares = posterior_share(Jf, lam)
    correlation = posterior_correlation(Jf, lam)
    partners = {free[k]: sorted(({"name": free[m], "corr": float(correlation[k, m])} for m in range(len(free))
                                 if m != k and abs(correlation[k, m]) >= partner_min),
                                key=lambda e: -abs(e["corr"])) for k in range(len(free))}
    d_post = {free[k]: float(shares[k]) for k in range(len(free_idx))}
    # In prior-sigma units: the prior rows sqrt(lambda) u give u a prior
    # variance of 1 / lambda, so the posterior sd relative to the prior's is
    # sqrt(lambda Sigma_ii) = sqrt(1 - d_post).
    posterior_sd = {free[k]: float(np.sqrt(max(1.0 - shares[k], 0.0))) for k in range(len(free_idx))}

    live = [i for i in free_idx if s[i] > 0]
    gamma_i = np.full(len(names), np.inf)
    if live:
        sub = np.linalg.pinv(gram[np.ix_(live, live)], rcond=1e-12, hermitian=True)
        for k, i in enumerate(live):
            gamma_i[i] = float(np.sqrt(max(sub[k, k], 0.0)))
    singular = np.linalg.svd(Jn[:, live], compute_uv=False) if live else np.zeros(0)
    raw_singular = np.linalg.svd(Jf, compute_uv=False)
    direction_share = raw_singular ** 2 / (raw_singular ** 2 + lam)

    frozen, final_gamma = _freeze(Jn, names, free_idx, s / s_max, sensitivity_rel_min, collinearity_max)

    rows = []
    for i, name in enumerate(names):
        rows.append({"name": name, "free": name in free, "sensitivity": float(s[i]),
                     "sensitivity_rel": float(s[i] / s_max), "data_share": float(d[i]),
                     "data_share_posterior": d_post.get(name), "posterior_sd_sigma": posterior_sd.get(name),
                     "prior_determined": bool((d_post[name] if name in d_post else d[i]) < PRIOR_DETERMINED_BELOW),
                     "collinearity": None if not np.isfinite(gamma_i[i]) else float(gamma_i[i]),
                     "partners": partners.get(name), "recommend_freeze": frozen.get(name)})
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if s[a] > 0 and s[b] > 0:
                pairs.append((abs(float(cos[a, b])), names[a], names[b], float(cos[a, b])))
    pairs.sort(reverse=True)
    gain_mass = []
    for a, name in enumerate(names):
        if name.startswith("sensor.ft.") and name.endswith(".gain"):
            best = [(abs(float(cos[a, b])), names[b], float(cos[a, b])) for b, other in enumerate(names)
                    if other.startswith("inertial.") and other.endswith(".mass") and s[a] > 0 and s[b] > 0]
            if best:
                top = max(best)
                gain_mass.append({"gain": name, "mass": top[1], "cos": top[2]})
    return {
        "n_rows": int(J.shape[0]), "n_parameters": len(names), "n_free": len(free), "prior_weight": lam,
        "parameters": rows,
        "singular_values_normalised": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else float("inf"),
        "collinearity_index": float(1.0 / singular[-1]) if len(singular) and singular[-1] > 0 else float("inf"),
        "directions_data_share": direction_share.tolist(),
        "n_directions_data_determined": int(np.sum(direction_share >= PRIOR_DETERMINED_BELOW)),
        "n_prior_determined_diagonal": int(sum(1 for i, n in enumerate(names) if n in free
                                               and d[i] < PRIOR_DETERMINED_BELOW)),
        "top_pairs": [{"a": a, "b": b, "cos": c} for _, a, b, c in pairs[:TOP_PAIRS]],
        "gain_mass_pairs": gain_mass,
        "recommended_frozen": [{"name": k, "reason": v} for k, v in frozen.items()],
        "collinearity_index_after_freeze": final_gamma,
        "n_prior_determined": int(sum(1 for r in rows if r["free"] and r["prior_determined"])),
        "n_prior_determined_posterior": int(sum(1 for v in d_post.values() if v < PRIOR_DETERMINED_BELOW)),
        "rules": {"sensitivity_rel_min": sensitivity_rel_min, "collinearity_max": collinearity_max,
                  "prior_determined_below": PRIOR_DETERMINED_BELOW, "partner_min": partner_min,
                  "columns": "d residual / d u, u = f(theta) / sigma (prior-sigma units), rows weighted as the loss"},
    }


def posterior_share(jacobian_u: np.ndarray, prior_weight: float) -> np.ndarray:
    """d_post per column: ``1 - lambda [(J^T J + lambda I)^-1]_ii``."""
    J = np.asarray(jacobian_u, dtype=float)
    if J.shape[1] == 0:
        return np.zeros(0)
    lam = float(prior_weight)
    covariance = np.linalg.inv(J.T @ J + lam * np.eye(J.shape[1]))
    return 1.0 - lam * np.diag(covariance)


def posterior_correlation(jacobian_u: np.ndarray, prior_weight: float) -> np.ndarray:
    """The posterior correlation matrix of the columns, from
    ``(J^T J + lambda I)^-1`` (the full set: every column is unknown)."""
    J = np.asarray(jacobian_u, dtype=float)
    if J.shape[1] == 0:
        return np.zeros((0, 0))
    covariance = np.linalg.inv(J.T @ J + float(prior_weight) * np.eye(J.shape[1]))
    sd = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sd, sd)


def labels(block: Mapping[str, Any], *, d_post_min: float = PRIOR_DETERMINED_BELOW) -> dict[str, dict[str, Any]]:
    """S2R_08 A-12': per free parameter of an :func:`analyse` block (on the
    full free set), ``calibrated`` iff d_post >= ``d_post_min``; the
    not-calibrated ones carry the reason and their partners. Nothing is
    recomputed: a frozen member would make its partners look known."""
    out = {}
    for row in block["parameters"]:
        if not row["free"]:
            continue
        d_post = row["data_share_posterior"]
        entry: dict[str, Any] = {"calibrated": bool(d_post >= d_post_min), "d_post": d_post}
        if not entry["calibrated"]:
            entry["reason"] = f"not separated by these data (d_post {d_post:.3f} < {d_post_min:g})"
            entry["partners"] = list(row.get("partners") or [])
        out[row["name"]] = entry
    return out


def freeze_plan(jacobian_at: Callable[[Mapping[str, float], Sequence[str]], np.ndarray], theta: Mapping[str, float],
                free: Sequence[str], freeze_value: Mapping[str, float], *, prior_weight: float,
                d_post_min: float = PRIOR_DETERMINED_BELOW) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """S2R_06 A-12, steps (2)-(4): freeze the lowest-d_post free parameter
    at ``freeze_value[name]`` and recompute until every remaining one has
    ``d_post >= d_post_min``. ``jacobian_at(theta, names)`` is the
    identifiability Jacobian (prior-sigma units, loss-weighted rows).
    Returns the frozen entries in order (name, the d_post that froze it, the
    value it had at the fit, the freeze value) and the theta with them
    frozen."""
    theta = dict(theta)
    free = list(free)
    frozen: list[dict[str, Any]] = []
    while free:
        shares = posterior_share(jacobian_at(theta, free), prior_weight)
        k = int(np.argmin(shares))
        if shares[k] >= d_post_min:
            break
        name = free.pop(k)
        frozen.append({"name": name, "d_post": float(shares[k]), "fitted": float(theta[name]),
                       "frozen_at": float(freeze_value[name]), "round_free": len(free) + 1})
        theta[name] = float(freeze_value[name])
    return frozen, theta


def _freeze(Jn: np.ndarray, names: Sequence[str], free_idx: Sequence[int], rel: np.ndarray, rel_min: float,
            gamma_max: float) -> tuple[dict[str, str], float]:
    frozen: dict[str, str] = {}
    keep = []
    for i in free_idx:
        if rel[i] < rel_min:
            frozen[names[i]] = f"sensitivity {rel[i]:.2e} of the largest (< {rel_min:g})"
        else:
            keep.append(i)
    gamma = float("inf")
    while keep:
        _, singular, vt = np.linalg.svd(Jn[:, keep], full_matrices=False)
        gamma = float(1.0 / singular[-1]) if singular[-1] > 0 else float("inf")
        if gamma <= gamma_max:
            break
        weakest = np.abs(vt[-1])
        drop = keep[int(np.argmax(weakest))]
        frozen[names[drop]] = f"collinearity: dominates the weakest direction (set gamma {gamma:.3g} > {gamma_max:g})"
        keep.remove(drop)
    return frozen, gamma


def format_partners(partners: Sequence[Mapping[str, Any]] | None, limit: int = 4) -> str:
    if not partners:
        return "–"
    text = ", ".join(f"`{e['name']}` {e['corr']:+.2f}" for e in partners[:limit])
    return text + (f" (+{len(partners) - limit})" if len(partners) > limit else "")


def to_markdown(block: dict[str, Any], *, title: str = "Identifiability") -> str:
    lines = [f"## {title}", "",
             f"{block['n_rows']} residual rows, {block['n_free']} free parameters "
             f"({block['n_parameters']} analysed); prior weight lambda = {block['prior_weight']:.3g}.",
             f"Condition number (column-normalised) {block['condition_number']:.3g}; collinearity index "
             f"{block['collinearity_index']:.3g} (after the recommended freeze: "
             f"{block['collinearity_index_after_freeze']:.3g}).",
             f"Data-determined directions (share >= {PRIOR_DETERMINED_BELOW}): "
             f"{block['n_directions_data_determined']} of {block['n_free']}. Prior-determined free parameters "
             f"(d_post < {PRIOR_DETERMINED_BELOW}, S2R_06 F-2): {block['n_prior_determined_posterior']} "
             f"(the diagonal d, a diagnostic, would say {block.get('n_prior_determined_diagonal', '–')}).", "",
             "| parameter | free | rel. sensitivity | d | d_post | post. sd (sigma) | collinearity | recommendation "
             "| partners (posterior corr) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in block["parameters"]:
        d_post = "" if row["data_share_posterior"] is None else f"{row['data_share_posterior']:.3f}"
        sd = "" if row["posterior_sd_sigma"] is None else f"{row['posterior_sd_sigma']:.3g}"
        gamma = "" if row["collinearity"] is None else f"{row['collinearity']:.3g}"
        flag = " **prior-determined**" if row["prior_determined"] else ""
        lines.append(f"| `{row['name']}` | {'yes' if row['free'] else 'no'} | {row['sensitivity_rel']:.2e} | "
                     f"{row['data_share']:.3f} | {d_post}{flag} | {sd} | {gamma} | {row['recommend_freeze'] or ''} | "
                     f"{format_partners(row.get('partners')) if row['prior_determined'] else ''} |")
    lines += ["", "**Most collinear pairs** (|cos| of the normalised columns):", "",
              "| a | b | cos |", "|---|---|---|"]
    lines += [f"| `{p['a']}` | `{p['b']}` | {p['cos']:+.4f} |" for p in block["top_pairs"]]
    if block["gain_mass_pairs"]:
        lines += ["", "**Sensor gain vs link mass** (S2R_04 R-3), the most collinear mass per gain:", "",
                  "| gain | mass | cos |", "|---|---|---|"]
        lines += [f"| `{p['gain']}` | `{p['mass']}` | {p['cos']:+.4f} |" for p in block["gain_mass_pairs"]]
    return "\n".join(lines) + "\n"


__all__ = ["COLLINEARITY_MAX", "PARTNER_MIN", "PRIOR_DETERMINED_BELOW", "SENSITIVITY_REL_MIN", "analyse",
           "format_partners", "freeze_plan", "labels", "posterior_correlation", "posterior_share", "to_markdown"]
