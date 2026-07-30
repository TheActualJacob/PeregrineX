"""FIXED evaluation harness for guided adversarial search. Do not modify.

Ground truth: proxy oracles fit to the 2,000-run ArduPlane SITL baseline
(sim/results/baseline.parquet). A search strategy sees a NOISY oracle
(sigma = 0.07, the measured SITL nondeterminism floor) and a budget of
BUDGET sequential queries over the unit hypercube [0,1]^8. It is scored on
DISTINCT failure and boundary cells discovered (truth from the noiseless
proxy), as a ratio to a Latin-hypercube reference given the identical budget
and seeds, averaged over PROXIES x SEEDS.

val_metric: mean(score_strategy / score_lhs). Higher is better. LHS == 1.0.

The dims mirror sim/docs/GOALS.md M4 (u-space -> physical is affine except
the conditional airspeed-failure pair):
  u0 wind_speed 0-15   u1 wind_dir 0-360   u2 turbulence 0-1
  u3 arspd_bias 0.7-1.3
  u4 arspd_fail_on (>0.5 = active)   u5 arspd_fail_dist 200-2500 (if active)
  u6 engine_mul 0.3-1.0              u7 gps_glitch 0-30
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASELINE = "/Users/jacob/Documents/AviationSim/sim/results/baseline.parquet"
DIMS = 8
BUDGET = 200
N_SEEDS = 10
NOISE_SIGMA = 0.07
FAIL_THR = 0.0
BOUNDARY_THR = 0.10
CELL_RES = 3          # cells per dim for distinctness scoring
TIME_BUDGET = 300     # seconds per trial, wall clock

RANGES = np.array([[0, 15], [0, 360], [0, 1], [0.7, 1.3],
                   [0, 1], [200, 2500], [0.3, 1.0], [0, 30]], dtype=float)


def _rows_to_u(df: pd.DataFrame) -> np.ndarray:
    u = np.empty((len(df), DIMS))
    cols = ["scn_wind_speed_mps", "scn_wind_dir_deg", "scn_turbulence",
            "scn_arspd_bias", None, "scn_arspd_fail_dist_m",
            "scn_engine_mul", "scn_gps_glitch_m"]
    fail_dist = df["scn_arspd_fail_dist_m"]
    active = fail_dist.notna().to_numpy()
    for j, c in enumerate(cols):
        if j == 4:
            u[:, j] = np.where(active, 0.75, 0.25)
        elif j == 5:
            v = fail_dist.fillna(RANGES[5].mean()).to_numpy()
            u[:, j] = (v - RANGES[5, 0]) / (RANGES[5, 1] - RANGES[5, 0])
        else:
            v = df[c].to_numpy(dtype=float)
            u[:, j] = (v - RANGES[j, 0]) / (RANGES[j, 1] - RANGES[j, 0])
    return np.clip(u, 0, 1)


def _fit_proxies(seed: int = 0):
    df = pd.read_parquet(BASELINE)
    df = df[df["scored"] == 1].dropna(subset=["safety_margin"])
    X, y = _rows_to_u(df), df["safety_margin"].to_numpy()
    y = np.clip(y, -1.5, 1.5)   # crash clamp region compressed; boundary intact
    proxies = {
        "gbm": HistGradientBoostingRegressor(max_iter=400, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=300, random_state=seed,
                                    n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed)),
    }
    fitted, r2 = {}, {}
    for name, m in proxies.items():
        cut = int(0.9 * len(X))
        m.fit(X[:cut], y[:cut])
        r2[name] = m.score(X[cut:], y[cut:])
        m_full = m.fit(X, y)
        fitted[name] = m_full
    return fitted, r2


_PROXIES, PROXY_R2 = _fit_proxies()


def _cells(u: np.ndarray) -> set:
    return {tuple(c) for c in np.minimum((u * CELL_RES).astype(int),
                                         CELL_RES - 1)}


def _score(u_queries: np.ndarray, truth: np.ndarray) -> float:
    fail = u_queries[truth < FAIL_THR]
    boundary = u_queries[np.abs(truth) < BOUNDARY_THR]
    return len(_cells(fail)) + len(_cells(boundary))


def _run_lhs(proxy, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = BUDGET
    u = (rng.permuted(np.tile(np.arange(n), (DIMS, 1)).T, axis=0)
         + rng.random((n, DIMS))) / n
    return _score(u, proxy.predict(u))


def evaluate_strategy(strategy_cls) -> float:
    """Fixed. Returns the one scalar the loop optimizes (higher = better).

    strategy_cls: class with __init__(dims, budget, rng) and methods
    ask() -> u (shape (DIMS,)) and tell(u, noisy_margin). Queried
    sequentially exactly BUDGET times per episode.
    """
    ratios = []
    for pname, proxy in _PROXIES.items():
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            strat = strategy_cls(DIMS, BUDGET, np.random.default_rng(seed))
            qs = np.empty((BUDGET, DIMS))
            for i in range(BUDGET):
                u = np.clip(np.asarray(strat.ask(), dtype=float), 0, 1)
                qs[i] = u
                true_m = float(proxy.predict(u[None, :])[0])
                strat.tell(u, true_m + rng.normal(0, NOISE_SIGMA))
            s = _score(qs, proxy.predict(qs))
            ref = _run_lhs(proxy, 5000 + seed)
            ratios.append(s / max(ref, 1e-9))
    return float(np.mean(ratios))


if __name__ == "__main__":
    print("proxy holdout R2:", {k: round(v, 3) for k, v in PROXY_R2.items()})
