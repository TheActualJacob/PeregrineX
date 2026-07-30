"""Integrity audit of the guided-search claim. Reads only; edits nothing.

Two ways the headline ratio could be softer than it looks:

1. INERT-DIM FARMING. When u4 < 0.5 the airspeed failure is off, so u5 is
   inert: two queries differing only in u5 are the same physical scenario, yet
   prepare.py counts them as distinct cells. A strategy that farms this gets
   credit LHS does not.
2. PROXY-SPECIFIC WIN. A ratio driven by one proxy family would not transfer to
   a real SITL farm.

Prints per-proxy ratios and the share of credited cells that are inert
duplicates, for the strategy and for LHS side by side.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

import prepare

HERE = Path(__file__).parent


def load_strategy(fname: str):
    spec = importlib.util.spec_from_file_location(fname[:-3], HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Strategy


def physical_cells(u: np.ndarray) -> set:
    """Cells, but collapsing u5 whenever the failure switch (u4) is off."""
    c = np.minimum((u * prepare.CELL_RES).astype(int), prepare.CELL_RES - 1)
    out = set()
    for row, uu in zip(c, u):
        t = list(row)
        if uu[4] < 0.5:      # failure inactive => u5 has no physical meaning
            t[5] = -1
        out.add(tuple(t))
    return out


def credited(u_q: np.ndarray, truth: np.ndarray):
    fail = u_q[truth < prepare.FAIL_THR]
    bnd = u_q[np.abs(truth) < prepare.BOUNDARY_THR]
    raw = len(prepare._cells(fail)) + len(prepare._cells(bnd))
    phys = len(physical_cells(fail)) + len(physical_cells(bnd))
    return raw, phys, len(fail), len(bnd)


def run_strategy(Strategy, proxy, seed: int):
    rng = np.random.default_rng(1000 + seed)
    strat = Strategy(prepare.DIMS, prepare.BUDGET, np.random.default_rng(seed))
    qs = np.empty((prepare.BUDGET, prepare.DIMS))
    for i in range(prepare.BUDGET):
        u = np.clip(np.asarray(strat.ask(), dtype=float), 0, 1)
        qs[i] = u
        t = float(proxy.predict(u[None, :])[0])
        strat.tell(u, t + rng.normal(0, prepare.NOISE_SIGMA))
    return qs, proxy.predict(qs)


def run_lhs(proxy, seed: int):
    rng = np.random.default_rng(5000 + seed)
    n = prepare.BUDGET
    u = (rng.permuted(np.tile(np.arange(n), (prepare.DIMS, 1)).T, axis=0)
         + rng.random((n, prepare.DIMS))) / n
    return u, proxy.predict(u)


def main() -> None:
    Strategy = load_strategy("train_best.py")
    seeds = range(5)   # audit subset; the metric itself uses 10
    hdr = (f"{'proxy':>6} {'raw':>7} {'phys':>7} {'failPts':>9} {'lhsFail':>8} "
           f"{'bndPts':>8} {'lhsBnd':>7} {'bndRatio':>9} {'failRatio':>10}")
    print(hdr)
    all_raw, all_phys = [], []
    for pname, proxy in prepare._PROXIES.items():
        rr, pr, sf, lf, sb, lb = [], [], [], [], [], []
        for s in seeds:
            qs, ts = run_strategy(Strategy, proxy, s)
            ql, tl = run_lhs(proxy, s)
            s_raw, s_phys, s_nf, s_nb = credited(qs, ts)
            l_raw, l_phys, l_nf, l_nb = credited(ql, tl)
            rr.append(s_raw / max(l_raw, 1e-9))
            pr.append(s_phys / max(l_phys, 1e-9))
            sf.append(s_nf); lf.append(l_nf)
            sb.append(s_nb); lb.append(l_nb)
        raw, phys = float(np.mean(rr)), float(np.mean(pr))
        all_raw.append(raw); all_phys.append(phys)
        print(f"{pname:>6} {raw:7.3f} {phys:7.3f} {np.mean(sf):9.1f} "
              f"{np.mean(lf):8.1f} {np.mean(sb):8.1f} {np.mean(lb):7.1f} "
              f"{np.mean(sb) / max(np.mean(lb), 1e-9):9.2f} "
              f"{np.mean(sf) / max(np.mean(lf), 1e-9):10.2f}")
    print(f"{'MEAN':>6} {np.mean(all_raw):7.3f} {np.mean(all_phys):7.3f}")
    print("\nbndPts/lhsBnd = boundary-band points (|margin|<0.10) found vs LHS.")
    print("If bndRatio >> failRatio, the advantage is boundary MAPPING, not")
    print("merely crash-finding -- the certification-relevant claim.")
    print("\nraw ratio  = the reported metric (cells as prepare.py counts them)")
    print("phys ratio = same, collapsing the inert u5 coordinate when u4<0.5")
    print("A large raw-minus-phys gap would mean the win is partly inert-dim")
    print("farming; a small gap means the discoveries are physically distinct.")
    print("Per-proxy spread shows whether the win is proxy-specific.")


if __name__ == "__main__":
    main()
