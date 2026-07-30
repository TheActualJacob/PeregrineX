"""EDITABLE search strategy. The agent loop improves this file only.

Iteration 2: ARD Gaussian-process surrogate + Expected Coverage Improvement.

The scored quantity is |distinct cells containing a true failure| +
|distinct cells containing a true boundary point|. A point with true margin in
(-0.10, 0) therefore earns DOUBLE credit, and a cell already credited earns
nothing more. So the right acquisition is not "minimise margin" and not plain
straddle -- it is expected coverage improvement (Malkomes et al. 2021) over the
cell partition:

    gain(u) = P(m(u) < 0) * (1 - q_fail[cell(u)])
            + P(|m(u)| < 0.1) * (1 - q_bnd[cell(u)])

with q_* the posterior probability that the cell is already credited by an
earlier query. Probabilities come from an ARD-RBF GP whose per-dimension
lengthscales are fitted by marginal likelihood, which matters here because the
landscape's sensitivity varies ~20x across the 8 dims.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

import prepare

# ---- EDITABLE HYPERPARAMETERS ----
WARMUP = 28            # LHS queries before the surrogate takes over
N_UNIF = 600           # uniform candidates (global exploration)
N_BASE = 90            # random bases for 1-D level-set sweeps
N_GRID = 11            # grid points per sweep
N_PERT = 500           # elite perturbations along flat directions
PERT = 0.5             # perturbation scale (x lengthscale)
ARD_EVERY = 25         # queries between ARD lengthscale re-fits
L_INIT = 0.45          # initial isotropic lengthscale (u-space)
L_MIN, L_MAX = 0.06, 12.0
SD_FLOOR = 0.02        # floor on posterior sd (guards over-confident probs)
SD_SCALE = 1.0         # scale on candidate posterior sd (<1 = optimistic)
BND_WEIGHT = 2.0       # weight on the double-credit boundary term
KERNEL = "rbf"         # "rbf" (smooth) or "exp" (Matern-1/2, rougher)
Y_CLIP = 0.28          # >0: clip observations to +-Y_CLIP before fitting, so
                       # the GP spends its capacity near the level set instead
                       # of on the +-1.5 crash-clamp plateaus
POLISH_ROUNDS = 3      # local refinement rounds on the acquisition surface
POLISH_N = 160         # perturbations per refinement round
POLISH_SCALE = 0.30    # initial refinement step (x lengthscale)

_CR = prepare.CELL_RES
_POW = _CR ** np.arange(prepare.DIMS)
_NCELL = _CR ** prepare.DIMS


class _GP:
    """Zero-mean ARD-RBF GP on standardised targets. n <= 200, so dense."""

    def __init__(self, dims: int):
        self.dims = dims
        self.w = np.full(dims, 1.0 / L_INIT ** 2)   # inverse squared lengthscales
        self.s2n = None                             # noise variance (std units)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _shape(d2):
        if KERNEL == "exp":
            return np.exp(-np.sqrt(np.maximum(d2, 0.0)))
        return np.exp(-0.5 * np.maximum(d2, 0.0))

    def _K(self, w):
        return self._shape(self.D @ w)

    def _factor(self, w, s2n):
        K = self._K(w)
        A = K + (s2n + 1e-8) * np.eye(len(K))
        return K, cho_factor(A, lower=True, check_finite=False)

    def _lml(self, w, s2n):
        try:
            K, c = self._factor(w, s2n)
        except np.linalg.LinAlgError:
            return -np.inf
        a = cho_solve(c, self.yn, check_finite=False)
        return -0.5 * float(self.yn @ a) - float(np.log(np.diag(c[0])).sum())

    # -- public ------------------------------------------------------------
    def set_data(self, X, y):
        self.X = X
        if Y_CLIP > 0:
            y = np.clip(y, -Y_CLIP, Y_CLIP)
        self.ym = float(y.mean())
        self.ys = float(y.std()) + 1e-9
        self.yn = (y - self.ym) / self.ys
        diff = X[:, None, :] - X[None, :, :]
        self.D = diff * diff                      # (n, n, d) per-dim sq dists
        if self.s2n is None:
            self.s2n = (prepare.NOISE_SIGMA / self.ys) ** 2

    def tune(self):
        """Coordinate ascent on the marginal likelihood (lengthscales + noise)."""
        best = self._lml(self.w, self.s2n)
        mults = (0.35, 0.6, 1.6, 3.0)
        for j in range(self.dims):
            l_j = 1.0 / np.sqrt(self.w[j])
            for m in mults:
                l_try = float(np.clip(l_j * m, L_MIN, L_MAX))
                w = self.w.copy()
                w[j] = 1.0 / l_try ** 2
                v = self._lml(w, self.s2n)
                if v > best:
                    best, self.w = v, w
                    l_j = l_try
        base = (prepare.NOISE_SIGMA / self.ys) ** 2
        for m in (0.5, 1.0, 2.0, 5.0):
            v = self._lml(self.w, base * m)
            if v > best:
                best, self.s2n = v, base * m

    def fit(self):
        self.K, self.c = self._factor(self.w, self.s2n)
        self.alpha = cho_solve(self.c, self.yn, check_finite=False)
        # posterior at the training points (denoised estimate of the truth)
        mu = self.K @ self.alpha
        V = cho_solve(self.c, self.K, check_finite=False)
        var = np.maximum(1.0 - np.einsum("ij,ij->j", self.K, V), 1e-9)
        self.mu_tr = mu * self.ys + self.ym
        self.sd_tr = np.sqrt(var) * self.ys

    def predict(self, C):
        Xw = self.X * self.w
        d = ((C * C) @ self.w)[:, None] + ((self.X * Xw).sum(1))[None, :] \
            - 2.0 * (C @ Xw.T)
        Ks = self._shape(d)
        mu = Ks @ self.alpha
        V = cho_solve(self.c, Ks.T, check_finite=False)
        var = np.maximum(1.0 - np.einsum("ij,ji->i", Ks, V), 1e-9)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys


def _codes(U):
    c = np.minimum((U * _CR).astype(int), _CR - 1)
    return c @ _POW


class Strategy:
    def __init__(self, dims: int, budget: int, rng: np.random.Generator):
        self.dims, self.budget, self.rng = dims, budget, rng
        self.X, self.y = [], []
        n = WARMUP
        self.warm = ((rng.permuted(np.tile(np.arange(n), (dims, 1)).T, axis=0)
                      + rng.random((n, dims))) / n).tolist()
        self.gp = _GP(dims)
        self.n_ard = 0

    # -- credited-cell bookkeeping ----------------------------------------
    def _credited(self):
        """Prob. that each cell is already credited for fail / boundary."""
        mu, sd = self.gp.mu_tr, np.maximum(self.gp.sd_tr, SD_FLOOR)
        pf = ndtr(-mu / sd)
        pb = ndtr((prepare.BOUNDARY_THR - mu) / sd) \
            - ndtr((-prepare.BOUNDARY_THR - mu) / sd)
        codes = _codes(np.asarray(self.X))
        lf = np.bincount(codes, weights=np.log1p(-np.clip(pf, 0, 1 - 1e-9)),
                         minlength=_NCELL)
        lb = np.bincount(codes, weights=np.log1p(-np.clip(pb, 0, 1 - 1e-9)),
                         minlength=_NCELL)
        return 1.0 - np.exp(lf), 1.0 - np.exp(lb)

    def _candidates(self) -> np.ndarray:
        """Uniform + 1-D level-set sweeps + elite perturbations.

        A random 7-dim configuration can almost always be pushed into the
        double-credit band by moving ONE coordinate (measured: 79% gbm, 80% rf,
        98% mlp), so sweeping a sensitivity-weighted coordinate over a grid
        gives a candidate pool far richer in band points than uniform sampling.
        Elite perturbations use per-dim steps proportional to the fitted
        lengthscale, which moves along the flat directions of the level set --
        changing the cell while holding the margin.
        """
        d = self.dims
        l = 1.0 / np.sqrt(self.gp.w)
        parts = [self.rng.random((N_UNIF, d))]

        if N_BASE and N_GRID:
            B = self.rng.random((N_BASE, d))
            p = 1.0 / l
            js = self.rng.choice(d, size=N_BASE, p=p / p.sum())
            g = (np.arange(N_GRID) + self.rng.random((N_BASE, N_GRID))) / N_GRID
            S = np.repeat(B, N_GRID, axis=0)
            S[np.arange(len(S)), np.repeat(js, N_GRID)] = g.ravel()
            parts.append(S)

        if N_PERT:
            hi, lo = prepare.BOUNDARY_THR, -2.0 * prepare.BOUNDARY_THR
            el = np.flatnonzero((self.gp.mu_tr < hi) & (self.gp.mu_tr > lo))
            if len(el):
                idx = self.rng.choice(el, size=N_PERT)
                step = PERT * np.minimum(l, 1.0)
                P = np.asarray(self.X)[idx] \
                    + self.rng.normal(0, 1, (N_PERT, d)) * step
                parts.append(np.clip(P, 0.0, 1.0))
        return np.concatenate(parts, 0)

    def ask(self) -> np.ndarray:
        if self.warm:
            return np.asarray(self.warm.pop())
        X, y = np.asarray(self.X), np.asarray(self.y)
        self.gp.set_data(X, y)
        if len(X) >= self.n_ard + ARD_EVERY:
            self.gp.tune()
            self.n_ard = len(X)
        self.gp.fit()
        qf, qb = self._credited()

        cand = self._candidates()
        acq = self._acq(cand, qf, qb)
        i = int(np.argmax(acq))
        u, best = cand[i], float(acq[i])

        # Local refinement of the acquisition surface. The random pool resolves
        # a swept coordinate to only 1/N_GRID ~ 0.09, which is coarse next to
        # the 0.10-wide boundary band, so argmax over the pool leaves value on
        # the table. Steps are lengthscale-scaled: large along inert dims,
        # small along the dims that actually move the margin.
        l = np.minimum(1.0 / np.sqrt(self.gp.w), 1.0)
        for r in range(POLISH_ROUNDS):
            step = POLISH_SCALE * (0.4 ** r) * l
            P = np.clip(u + self.rng.normal(0, 1, (POLISH_N, self.dims)) * step,
                        0.0, 1.0)
            a = self._acq(P, qf, qb)
            j = int(np.argmax(a))
            if a[j] > best:
                u, best = P[j], float(a[j])
        return u

    def _acq(self, cand, qf, qb):
        mu, sd = self.gp.predict(cand)
        sd = np.maximum(sd * SD_SCALE, SD_FLOOR)
        p_fail = ndtr(-mu / sd)
        p_bnd = ndtr((prepare.BOUNDARY_THR - mu) / sd) \
            - ndtr((-prepare.BOUNDARY_THR - mu) / sd)
        cc = _codes(cand)
        return p_fail * (1.0 - qf[cc]) + BND_WEIGHT * p_bnd * (1.0 - qb[cc])

    def tell(self, u: np.ndarray, noisy_margin: float) -> None:
        self.X.append(np.asarray(u))
        self.y.append(noisy_margin)


if __name__ == "__main__":
    t0 = time.time()
    metric = prepare.evaluate_strategy(Strategy)
    print("---")
    print(f"val_metric:       {metric:.6f}")
    print(f"training_seconds: {time.time() - t0:.1f}")
    print(f"proxy_r2:         {prepare.PROXY_R2}")
