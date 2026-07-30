# Guided adversarial search — research loop

## Objective

Maximize `val_metric` printed by `train.py`: the sample-efficiency ratio of the
search strategy vs Latin-hypercube sampling at finding DISTINCT failure and
boundary cells of a simulated autopilot's safety-margin landscape, under query
noise, averaged over 3 proxy oracles x 10 seeds. LHS scores 1.0 by
construction. Higher is better.

## Loop

1. Run: `/Users/jacob/Documents/AviationSim/sim/venv/bin/python train.py`
   (cwd: `/Users/jacob/Documents/AviationSim/search_loop/`)
2. Read `val_metric` from the summary block (crash or NaN → status `crash`).
3. Append one row to `results.tsv`:
   `iteration<TAB>val_metric<TAB>seconds<TAB>status<TAB>one-line description`
4. Keep the edit if `val_metric` improved on the best so far; otherwise revert
   `train.py` to the best version (keep a copy at `train_best.py`).
5. Edit `train.py` (ONLY this file) with the next idea. Repeat.

## Rules

- `prepare.py` is FIXED. Never edit it. If it looks wrong, note it in
  `results.tsv` description and continue.
- Do not change the summary block format in `train.py`.
- One trial must finish within `prepare.TIME_BUDGET` (300 s) — if slower,
  reduce N_CAND / ensemble size.
- Never `git commit`.
- Web research is allowed and encouraged: batch Bayesian optimization,
  TuRBO/trust regions, level-set estimation (the true task is level-set
  mapping of margin=0, not minimization), Thompson sampling, entropy search,
  CMA-ES variants, diversity-driven acquisition (novelty/DPP).
- The strategy must remain a sequential ask/tell class named `Strategy`
  with the constructor signature `(dims, budget, rng)`.
- Beware overfitting to one proxy family: the metric averages over GBM, RF,
  and MLP proxies — wins must generalize across all three.

## Idea backlog (start here, in rough order)

1. Tune acquisition weights + WARMUP (cheap sweep).
2. Straddle/level-set acquisition (LSE): target |mu| ~ 1.96*sd band.
3. Batch diversity: penalize candidates near previous queries in u-space
   (min-distance), not just cell novelty.
4. Trust-region restarts (TuRBO-style) around discovered failure clusters.
5. Conditional-dim awareness: u4<0.5 makes u5 inert — avoid wasting distinct
   cells on the dead coordinate.
6. Thompson sampling from the bootstrap ensemble instead of argmax(acq).
7. Adaptive noise handling: repeat-query averaging near the boundary.
