"""Base parameters, Zollo's categorization and the least-squares solve.

The stacked regressor ``[W | T]`` is never held in memory: each bag's rows
are folded into the triangular factor ``R`` of ``[W | T]`` (incremental QR),
which keeps every quantity the identification needs exactly --
``W^T W = R_W^T R_W``, the column norms, the least-squares solution and its
residual.

Zollo finds the independent columns by Gauss-Jordan elimination (eqs.
39-41): it keeps, in column order, every column not spanned by the ones
kept before it.  :func:`select_base` does the same with a Gram-Schmidt sweep
over the RMS-scaled columns of ``R`` and an explicit tolerance, which is the
numerically robust equivalent.  Each original parameter is then

* ``unidentifiable``: its column is zero (it never acts on the rows);
* ``identifiable``: independent, and no dependent column folds into it;
* ``combination``: dependent, or independent with dependent columns folded
  into it -- identifiable only through the base parameter
  ``Pi*_i = Pi_i + sum_d Gamma_id Pi_d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

CATEGORIES = ("identifiable", "combination", "unidentifiable")


class Accumulator:
    """The triangular factor of the stacked ``[W | T]``, one bag at a time."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.R = np.zeros((0, self.size + 1))
        self.rows = 0

    def add(self, W: np.ndarray, target: np.ndarray, columns: Sequence[int] | None = None,
            weights: np.ndarray | None = None) -> None:
        """Fold ``rows`` of ``W`` (over layout positions ``columns``) and ``target`` in."""
        W = np.atleast_2d(np.asarray(W, dtype=float))
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(W) == 0:
            return
        block = np.zeros((len(W), self.size + 1))
        block[:, np.arange(self.size) if columns is None else np.asarray(columns, dtype=int)] = W
        block[:, -1] = target
        if weights is not None:
            block *= np.asarray(weights, dtype=float).reshape(-1, 1)
        if not np.isfinite(block).all():
            raise ValueError("non-finite regressor rows")
        self.R = np.linalg.qr(np.vstack([self.R, block]), mode="r")
        self.rows += len(W)

    @property
    def regressor(self) -> np.ndarray:
        """``R_W``: ``||W x - T|| = ||R_W x - z||`` for every ``x``."""
        return self._square()[:, :-1]

    @property
    def target(self) -> np.ndarray:
        return self._square()[:, -1]

    def _square(self) -> np.ndarray:
        R = self.R
        if len(R) < self.size + 1:
            R = np.vstack([R, np.zeros((self.size + 1 - len(R), self.size + 1))])
        return R


@dataclass
class Base:
    """The independent columns and how the others fold into them."""

    independent: np.ndarray
    dependent: np.ndarray
    zero: np.ndarray
    relation: np.ndarray        # [r, d], physical units: Pi* = Pi[I] + relation @ Pi[D]
    categories: list[str]
    column_rms: np.ndarray
    condition_scaled: float
    condition_raw: float
    tolerance: float

    @property
    def rank(self) -> int:
        return int(len(self.independent))

    def base_of(self, full: np.ndarray) -> np.ndarray:
        """The base vector of a full parameter vector (``nan`` entries count as unknown)."""
        full = np.asarray(full, dtype=float)
        dependent = full[self.dependent]
        out = full[self.independent].copy()
        if len(self.dependent):
            used = self.relation != 0.0
            out = out + np.where(used, self.relation * np.where(np.isnan(dependent), 0.0, dependent)[None, :],
                                 0.0).sum(axis=1)
            unknown = (used & np.isnan(dependent)[None, :]).any(axis=1)
            out[unknown] = np.nan
        return out

    def full_of(self, base: np.ndarray, size: int) -> np.ndarray:
        """A full vector equivalent to ``base`` on the rows: dependent and zero columns set to 0."""
        full = np.zeros(size)
        full[self.independent] = np.asarray(base, dtype=float)
        return full

    def expression(self, i: int, names: Sequence[str], *, threshold: float = 1e-9) -> str:
        """``Pi*_i`` written out over the original parameters."""
        terms = [names[self.independent[i]]]
        for k, d in enumerate(self.dependent):
            coefficient = self.relation[i, k]
            if abs(coefficient) > threshold * max(1.0, np.abs(self.relation[i]).max(initial=0.0)):
                terms.append(f"{coefficient:+.6g}*{names[d]}")
        return " ".join(terms)

    def describe(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "rank": self.rank, "n_parameters": len(names), "tolerance": self.tolerance,
            "condition_scaled": self.condition_scaled, "condition_raw": self.condition_raw,
            "independent": [names[i] for i in self.independent],
            "dependent": [names[i] for i in self.dependent],
            "unidentifiable": [names[i] for i in self.zero],
            "categories": dict(zip(names, self.categories)),
            "base_expressions": [self.expression(i, names) for i in range(self.rank)],
        }


def select_base(acc: Accumulator, *, tolerance: float = 1e-6, zero_tolerance: float = 1e-10) -> Base:
    """Zollo's elimination on the RMS-scaled columns, with explicit tolerances.

    A column is kept when the part of it outside the span of the columns kept
    before it is larger than ``tolerance`` (relative; the scaled columns have
    unit norm).  A column whose RMS is below ``zero_tolerance`` times the
    largest is unidentifiable.
    """
    if acc.rows == 0:
        raise ValueError("no rows accumulated")
    RW = acc.regressor
    norms = np.linalg.norm(RW, axis=0)
    rms = norms / np.sqrt(acc.rows)
    zero = norms <= zero_tolerance * max(norms.max(), np.finfo(float).tiny)
    scaled = np.where(zero[None, :], 0.0, RW / np.where(zero, 1.0, norms)[None, :])
    basis = np.zeros((len(RW), 0))
    independent, dependent = [], []
    for k in range(RW.shape[1]):
        if zero[k]:
            continue
        column = scaled[:, k]
        residual = column - basis @ (basis.T @ column)
        residual = residual - basis @ (basis.T @ residual)
        size = float(np.linalg.norm(residual))
        if size > tolerance:
            basis = np.column_stack([basis, residual / size])
            independent.append(k)
        else:
            dependent.append(k)
    independent = np.asarray(independent, dtype=int)
    dependent = np.asarray(dependent, dtype=int)
    if len(dependent):
        beta = np.linalg.lstsq(scaled[:, independent], scaled[:, dependent], rcond=None)[0]
        relation = beta * norms[dependent][None, :] / norms[independent][:, None]
    else:
        beta = relation = np.zeros((len(independent), 0))
    categories = []
    folded = (np.abs(beta) > 1e-6).any(axis=1) if beta.size else np.zeros(len(independent), dtype=bool)
    position = {int(k): i for i, k in enumerate(independent)}
    for k in range(RW.shape[1]):
        if zero[k]:
            categories.append("unidentifiable")
        elif k in position:
            categories.append("combination" if folded[position[k]] else "identifiable")
        else:
            categories.append("combination")
    singular_scaled = np.linalg.svd(scaled[:, independent], compute_uv=False)
    singular_raw = np.linalg.svd(RW[:, independent], compute_uv=False)
    return Base(independent, dependent, np.flatnonzero(zero), relation, categories, rms,
                float(singular_scaled[0] / singular_scaled[-1]), float(singular_raw[0] / singular_raw[-1]),
                float(tolerance))


@dataclass
class Solution:
    base: np.ndarray           # Pi*, physical units
    std: np.ndarray            # its standard deviation (LS covariance)
    rss: float
    rows: int
    residual_rms: float
    ridge: float
    full: np.ndarray | None = None   # a constrained solve's full vector (consistent inertials), else None


def solve(acc: Accumulator, base: Base, *, ridge: float = 0.0, prior: np.ndarray | None = None) -> Solution:
    """Least squares for the base vector (Zollo eq. 42), optionally ridged toward ``prior``.

    ``ridge`` is dimensionless: it weighs ``||x - prior||`` in the scaled
    space, where every base column has unit norm over all rows.
    """
    RW, z = acc.regressor, acc.target
    norms = base.column_rms[base.independent] * np.sqrt(acc.rows)
    A = RW[:, base.independent] / norms[None, :]
    b = z
    if ridge > 0.0:
        if prior is None or not np.isfinite(prior).all():
            raise ValueError("a ridge needs a finite prior base vector")
        A = np.vstack([A, ridge * np.eye(A.shape[1])])
        b = np.concatenate([z, ridge * np.asarray(prior, dtype=float) * norms])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rss = float(np.sum((RW[:, base.independent] / norms[None, :] @ x - z) ** 2))
    dof = max(acc.rows - base.rank, 1)
    sigma2 = rss / dof
    gram = A.T @ A
    covariance = sigma2 * np.linalg.pinv(gram)
    return Solution(x / norms, np.sqrt(np.clip(np.diag(covariance), 0.0, None)) / norms, rss, acc.rows,
                    float(np.sqrt(rss / acc.rows)), float(ridge))
