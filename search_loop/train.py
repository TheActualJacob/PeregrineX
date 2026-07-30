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
from sklearn.ensemble import RandomForestRegressor

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
SURROGATE = "gp"       # "gp" or "forest". Measured on 200 noisy points, a
                       # random forest beats the ARD-GP on ALL three proxies
                       # (RMSE gbm .130 vs .165, rf .094 vs .121, mlp .182 vs
                       # .183) -- the margin surface has axis-aligned
                       # threshold/regime structure that a stationary RBF
                       # kernel smooths away. The GP is retained either way as
                       # the source of ARD lengthscales for candidate
                       # generation, since a forest gives no natural metric.
N_TREES = 60
FOREST_EVERY = 4       # queries between forest refits
FOREST_SD_K = 1.0      # calibration on the per-tree spread used as sigma
ADD_MIX = 0.0          # 0 = pure product (fully interacting) RBF kernel,
                       # 1 = purely additive (main effects only). With only 200
                       # noisy points in 8-D a product kernel is starved; an
                       # additive kernel estimates per-dim main effects from all
                       # 200 points at once. The mixture keeps some interaction.
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
        if ADD_MIX <= 0.0:
            return self._shape(self.D @ w)
        E = self._shape(self.D * w)                    # (n, n, d) per-dim
        return (1.0 - ADD_MIX) * E.prod(-1) + ADD_MIX * E.mean(-1)

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
        if ADD_MIX > 0.0:
            add = np.zeros_like(Ks)
            for j in range(self.dims):
                dj = C[:, j][:, None] - self.X[None, :, j]
                add += self._shape(self.w[j] * dj * dj)
            Ks = (1.0 - ADD_MIX) * Ks + ADD_MIX * (add / self.dims)
        mu = Ks @ self.alpha
        V = cho_solve(self.c, Ks.T, check_finite=False)
        var = np.maximum(1.0 - np.einsum("ij,ji->i", Ks, V), 1e-9)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys


class _Forest:
    """Bootstrap random forest. Mean over trees = mu, spread over trees = sd,
    out-of-bag predictions = denoised estimate of the truth at queried points."""

    def fit(self, X, y):
        self.m = RandomForestRegressor(
            n_estimators=N_TREES, min_samples_leaf=2, oob_score=True,
            n_jobs=1, random_state=0, bootstrap=True)
        self.m.fit(X, y)
        oob = np.asarray(self.m.oob_prediction_, dtype=float)
        ins = self.m.predict(X)
        self.mu_tr = np.where(np.isfinite(oob), oob, ins)
        per = np.stack([t.predict(X) for t in self.m.estimators_])
        self.sd_tr = per.std(0) * FOREST_SD_K

    def predict(self, C):
        per = np.stack([t.predict(C) for t in self.m.estimators_])
        return per.mean(0), per.std(0) * FOREST_SD_K


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
        self.forest = _Forest() if SURROGATE == "forest" else None
        self.n_ard = self.n_fit = 0
        self.mu_tr = self.sd_tr = None

    # -- credited-cell bookkeeping ----------------------------------------
    def _credited(self):
        """Prob. that each cell is already credited for fail / boundary."""
        mu, sd = self.mu_tr, np.maximum(self.sd_tr, SD_FLOOR)
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
            el = np.flatnonzero((self.mu_tr < hi) & (self.mu_tr > lo))
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
        if self.forest is None or len(X) >= self.n_ard + ARD_EVERY:
            # the GP is refit every step in gp mode; in forest mode it is kept
            # alive only to supply ARD lengthscales for candidate generation
            self.gp.set_data(X, y)
            if len(X) >= self.n_ard + ARD_EVERY:
                self.gp.tune()
                self.n_ard = len(X)
            self.gp.fit()
        if self.forest is None:
            self.mu_tr, self.sd_tr = self.gp.mu_tr, self.gp.sd_tr
        else:
            if len(X) >= self.n_fit + FOREST_EVERY or self.mu_tr is None:
                self.forest.fit(X, np.clip(y, -Y_CLIP, Y_CLIP)
                                if Y_CLIP > 0 else y)
                self.n_fit = len(X)
            # the forest is refit only every FOREST_EVERY queries, so points
            # observed since then are represented by their own raw observation
            # (an unbiased, if noisy, estimate of the truth there)
            k, n = len(self.forest.mu_tr), len(X)
            self.mu_tr, self.sd_tr = self.forest.mu_tr, self.forest.sd_tr
            if n > k:
                ex = np.clip(y[k:], -Y_CLIP, Y_CLIP) if Y_CLIP > 0 else y[k:]
                self.mu_tr = np.concatenate([self.mu_tr, ex])
                self.sd_tr = np.concatenate(
                    [self.sd_tr, np.full(n - k, prepare.NOISE_SIGMA)])
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
        mu, sd = (self.forest if self.forest is not None else self.gp).predict(cand)
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
