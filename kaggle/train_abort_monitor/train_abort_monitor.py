"""Train the abort/go-around monitor on the week-1 SITL evidence.

Input dataset: jacobruan1/aviationsim-week1-evidence-v3
  trajectories.parquet -- every run's ~2 Hz trajectory sidecar, concatenated;
                          run_id column already fixed to the true run id.
  baseline.parquet     -- 2,000-run LHS baseline: labels (pass, safety_margin,
                          status, go_around) and flattened scn_* scenario dims.
  shakeout500.parquet  -- 500-run shakeout (seed 1000): out-of-sample check.

Outputs (/kaggle/working):
  abort_monitor.pt    -- GRU weights + feature scaler + config (torch.save)
  abort_monitor.onnx  -- same net, dynamic time axis, for the Mac companion
  eval.json           -- held-out ROC-AUC by lead time + threshold sweep

The monitor sees only what a companion computer sees in flight (telemetry-
derived features). Scenario parameters are never inputs -- the deployed
monitor will not know the wind.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DATA = Path("/kaggle/input/aviationsim-week1-evidence-v3")
WORK = Path("/kaggle/working")

HZ = 2.0                 # sidecar sample rate
WINDOW_S = 90.0          # final N seconds before touchdown/end of data
T = int(WINDOW_S * HZ)   # fixed sequence length, oldest-first
BASE_FEATURES = ["alt_agl_m", "airspeed_mps", "vz_mps", "along_m", "cross_m"]
DIFF_FEATURES = ["alt_agl_m", "along_m", "cross_m"]  # per-second finite diffs
N_FEATURES = len(BASE_FEATURES) + len(DIFF_FEATURES)
LEAD_BUCKETS_S = [5, 10, 20, 30, 45, 60]

torch.manual_seed(42)
np.random.seed(42)


def load_sidecars() -> dict[str, pd.DataFrame]:
    traj = pd.read_parquet(DATA / "trajectories.parquet")
    # deduped dataset version = 2,997,267 rows; v1 (26 interleaved retried
    # runs) = 3,021,514. If this prints the latter, the kernel got stale data.
    print(f"trajectories.parquet rows: {len(traj):,}")
    return {run_id: df for run_id, df in traj.groupby("run_id")}


def labels_from(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df[df["scored"] == 1].copy()
    # A commanded go-around is the behavior this monitor exists to produce;
    # those runs are neither positive nor negative examples. Drop them.
    df = df[~df["go_around"].fillna(False).astype(bool)]
    df["fail"] = ~df["pass"].fillna(0).astype(bool)
    return df.set_index("run_id")[["fail", "safety_margin", "status"]]


def sequence_for(traj: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """(T, F) feature grid over the last WINDOW_S seconds, plus per-step lead
    time to touchdown (or end of data). Oldest-first; front-padded with the
    first row so the GRU warm-starts from a real state, with lead = +inf there
    masked out of the loss by the caller via the valid mask (lead >= 0)."""
    traj = traj.sort_values("t_s")
    t = traj["t_s"].to_numpy()
    if len(t) < 10:
        return None
    td = traj["touchdown_t_s"].dropna()
    t_end = float(td.iloc[0]) if len(td) else float(t[-1])
    grid = np.arange(t_end - WINDOW_S, t_end, 1.0 / HZ)

    feats = np.empty((T, N_FEATURES), dtype=np.float32)
    col = 0
    for name in BASE_FEATURES:
        v = np.interp(grid, t, traj[name].ffill().fillna(0.0).to_numpy())
        feats[:, col] = v
        col += 1
    for name in DIFF_FEATURES:
        v = np.interp(grid, t, traj[name].ffill().fillna(0.0).to_numpy())
        feats[:, col] = np.gradient(v, 1.0 / HZ)
        col += 1
    lead = (t_end - grid).astype(np.float32)  # seconds until touchdown/end
    valid = (grid >= t[0]).astype(np.float32)  # inside recorded flight
    return feats, np.where(valid > 0, lead, -1.0)


def build(split_ids, sidecars, labels):
    xs, leads, ys, kept = [], [], [], []
    for rid in split_ids:
        if rid not in sidecars or rid not in labels.index:
            continue
        seq = sequence_for(sidecars[rid])
        if seq is None:
            continue
        xs.append(seq[0])
        leads.append(seq[1])
        ys.append(float(labels.loc[rid, "fail"]))
        kept.append(rid)
    return (np.stack(xs), np.stack(leads), np.asarray(ys, np.float32), kept)


class Monitor(nn.Module):
    """Per-step P(this run ends in failure) from telemetry alone."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, T, F) -> (B, T) logits
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank AUC without sklearn."""
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pick_device() -> str:
    """CUDA only if a cuDNN GRU actually runs on it — Kaggle sometimes hands
    out a GPU arch the preinstalled torch wheel has no kernels for (P100/sm_60),
    which dies at .to(device) with cudaErrorNoKernelImageForDevice."""
    if torch.cuda.is_available():
        try:
            probe = nn.GRU(2, 2, batch_first=True).to("cuda")
            probe(torch.zeros(1, 4, 2, device="cuda"))
            torch.cuda.synchronize()
            return "cuda"
        except Exception as exc:  # noqa: BLE001
            print(f"cuda present but unusable ({exc}); using cpu")
    return "cpu"


def main() -> None:
    t0 = time.time()
    device = pick_device()
    print(f"device: {device}")

    sidecars = load_sidecars()
    base = labels_from(DATA / "baseline.parquet")
    shake = labels_from(DATA / "shakeout500.parquet")
    print(f"sidecars: {len(sidecars)}, baseline rows: {len(base)}, "
          f"shakeout rows: {len(shake)}")

    ids = [r for r in base.index if r in sidecars]
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    n_val = max(1, len(ids) // 5)
    val_ids, train_ids = ids[:n_val], ids[n_val:]

    x_tr, lead_tr, y_tr, _ = build(train_ids, sidecars, base)
    x_va, lead_va, y_va, _ = build(val_ids, sidecars, base)
    x_sh, lead_sh, y_sh, _ = build(list(shake.index), sidecars, shake)
    print(f"train {len(x_tr)} (fail {y_tr.mean():.1%}), "
          f"val {len(x_va)} (fail {y_va.mean():.1%}), "
          f"shakeout {len(x_sh)} (fail {y_sh.mean():.1%})")

    mean = x_tr.reshape(-1, N_FEATURES).mean(0)
    std = x_tr.reshape(-1, N_FEATURES).std(0) + 1e-6

    def tensors(x, lead, y):
        return (torch.tensor((x - mean) / std, device=device),
                torch.tensor(lead, device=device),
                torch.tensor(y, device=device))

    xt, lt, yt = tensors(x_tr, lead_tr, y_tr)
    xv, lv, yv = tensors(x_va, lead_va, y_va)

    model = Monitor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    batch = 128
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            logits = model(xt[idx])
            mask = (lt[idx] >= 0).float()
            per_step = bce(logits, yt[idx, None].expand_as(logits))
            loss = (per_step * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        if epoch % 5 == 4 or epoch == 0:
            model.eval()
            with torch.no_grad():
                s = torch.sigmoid(model(xv))[:, -1].cpu().numpy()
            print(f"epoch {epoch + 1:2d} loss {total / len(xt):.4f} "
                  f"val AUC@last {auc(y_va, s):.3f}")

    # ---- evaluation: AUC as a function of warning lead time -----------------
    model.eval()
    report: dict = {"train_runs": len(x_tr), "val_runs": len(x_va),
                    "shakeout_runs": len(x_sh), "auc_by_lead_s": {},
                    "shakeout_auc_by_lead_s": {}, "threshold_sweep": []}
    for name, (x, lead, y) in [("auc_by_lead_s", (xv, lv, yv)),
                               ("shakeout_auc_by_lead_s", tensors(x_sh, lead_sh, y_sh))]:
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        leads = lead.cpu().numpy()
        ys = y.cpu().numpy()
        for lb in LEAD_BUCKETS_S:
            # score = the monitor's belief at the last step at least lb seconds out
            ok = leads >= lb
            score = np.array([p[m].max() if m.any() else 0.0
                              for p, m in zip(probs, ok)])
            report[name][str(lb)] = round(auc(ys, score), 4)
    with torch.no_grad():
        pv = torch.sigmoid(model(xv)).cpu().numpy()
    for thr in np.arange(0.3, 0.95, 0.05):
        fired = np.array([((p > thr) & (l >= 10)).any()
                          for p, l in zip(pv, lv.cpu().numpy())])
        tp = (fired & (y_va > 0.5)).sum(); fp = (fired & (y_va < 0.5)).sum()
        report["threshold_sweep"].append({
            "threshold": round(float(thr), 2),
            "abort_rate": round(float(fired.mean()), 4),
            "caught_failures": round(float(tp / max(y_va.sum(), 1)), 4),
            "false_abort_rate": round(float(fp / max((y_va < 0.5).sum(), 1)), 4),
        })
    print(json.dumps(report, indent=2))
    (WORK / "eval.json").write_text(json.dumps(report, indent=2))

    torch.save({"state_dict": model.state_dict(),
                "feature_mean": mean, "feature_std": std,
                "features": BASE_FEATURES + [f"d_{f}" for f in DIFF_FEATURES],
                "hz": HZ, "window_s": WINDOW_S}, WORK / "abort_monitor.pt")
    model.cpu().eval()
    example = torch.zeros(1, T, N_FEATURES)
    try:
        traced = torch.jit.trace(model, example)
        traced.save(str(WORK / "abort_monitor.torchscript.pt"))
        print("torchscript export ok")
    except Exception as exc:  # noqa: BLE001 - .pt is already saved
        print(f"torchscript export failed (non-fatal): {exc}")
    for kwargs in ({"dynamo": False}, {}):
        # dynamo=False uses the legacy exporter, which needs no onnxscript
        # (not installed, and this kernel has no internet to fetch it).
        try:
            torch.onnx.export(
                model, example, str(WORK / "abort_monitor.onnx"),
                input_names=["telemetry"], output_names=["fail_logit"],
                dynamic_axes={"telemetry": {0: "batch", 1: "time"},
                              "fail_logit": {0: "batch", 1: "time"}},
                **kwargs)
            print(f"onnx export ok ({kwargs or 'default'})")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"onnx export {kwargs} failed (non-fatal): {exc}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
