"""Failure map for PX4's neural controller vs the classical stack.

    python analysis_px4/neural_failure_map.py --results <results.jsonl> \
        --out analysis_px4/report

Reads the scenario-campaign JSONL produced by the Kaggle kernel
`px4_neural_scenarios` (one row per flight: scenario dims, arm, oracle
metrics, margin) and renders the exhibits:

  1. boundary.png    -- mass step x thrust ceiling, colored by margin, one
                        panel per control arm. Where does each controller
                        stop holding?
  2. severity.png    -- lateral drift during the adversarial event. The
                        shakeout finding: when both arms lose altitude, the
                        network ALSO loses lateral control by an order of
                        magnitude more than PID.
  3. marginals.png   -- per-dimension failure rate, both arms.
  4. paired.png      -- per-scenario margin difference (neural - classical),
                        the controlled comparison.
  5. REPORT.md       -- headline numbers, written from the data.

Palette is imported from the existing report so hues never mean two things
across the deck. safety margin has a meaningful zero, so it stays diverging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from analysis.failure_map import (  # noqa: E402
    AXIS, CRITICAL, FIG_DPI, GRID, INK, INK_MUTED, INK_SECONDARY,
    MARGIN_CMAP, SERIES_BLUE, SURFACE,
)

DIMS = {
    "mass_mul": "mass step (x nominal)",
    "tmax_mul": "thrust ceiling (x nominal)",
    "ixx_mul": "inertia (x nominal)",
    "kdv": "linear drag",
}
ARMS = ["classical", "neural"]
ARM_LABEL = {"classical": "classical (PID/Hold)",
             "neural": "neural (mc_nn_control)"}


def _style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, color=INK, fontsize=11, fontweight="bold", loc="left",
                 pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, color=INK_MUTED,
                fontsize=8.5, va="bottom")


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    scored = df[df["status"].isin(["held", "crashed"])].copy()
    scored["failed"] = scored["margin"] < 0
    return df, scored


def fig_boundary(df: pd.DataFrame, out: Path) -> None:
    norm = TwoSlopeNorm(vmin=min(-1.0, df.margin.min()), vcenter=0.0,
                        vmax=max(1.0, df.margin.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, arm in zip(axes, ARMS):
        d = df[df.arm == arm]
        _style(ax)
        sc = ax.scatter(d.mass_mul, d.tmax_mul, c=d.margin, cmap=MARGIN_CMAP,
                        norm=norm, s=34, edgecolor=SURFACE, linewidth=0.4,
                        zorder=2)
        crash = d[d.status == "crashed"]
        ax.scatter(crash.mass_mul, crash.tmax_mul, facecolors="none",
                   edgecolor=INK, linewidth=0.7, s=64, zorder=3,
                   label="crashed")
        ax.set_xlabel(DIMS["mass_mul"])
        _title(ax, ARM_LABEL[arm],
               f"{len(d)} runs, {100 * (d.margin < 0).mean():.0f}% fail")
        if arm == ARMS[0]:
            ax.set_ylabel(DIMS["tmax_mul"])
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
                      loc="lower left")
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("safety margin (0 = limit)", color=INK_SECONDARY, fontsize=8.5)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)
    fig.savefig(out, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_severity(df: pd.DataFrame, out: Path) -> tuple[float, float]:
    """The money figure: lateral drift, split by whether the run crashed."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    stats = {}
    for ax, crashed in zip(axes, [False, True]):
        _style(ax, grid_axis="y")
        sub = df[(df.status == "crashed") == crashed]
        data = [sub[sub.arm == a].max_dev_m.dropna().to_numpy() for a in ARMS]
        parts = ax.boxplot(data, tick_labels=[ARM_LABEL[a] for a in ARMS],
                           patch_artist=True, widths=0.5, showfliers=False,
                           medianprops=dict(color=INK, linewidth=1.4))
        for patch, color in zip(parts["boxes"], [SERIES_BLUE, CRITICAL]):
            patch.set_facecolor(color)
            patch.set_alpha(0.30)
            patch.set_edgecolor(color)
        for i, arr in enumerate(data, start=1):
            if len(arr):
                ax.scatter(np.full(len(arr), i)
                           + np.random.default_rng(0).normal(0, 0.045, len(arr)),
                           arr, s=12, color=INK_SECONDARY, alpha=0.45, zorder=3)
        ax.set_ylabel("max lateral drift during event (m)")
        med = [float(np.median(a)) if len(a) else float("nan") for a in data]
        stats["crashed" if crashed else "held"] = med
        _title(ax, "crash scenarios" if crashed else "recovered scenarios",
               f"median drift {med[0]:.2f} m vs {med[1]:.2f} m"
               if not any(np.isnan(med)) else "")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    c = stats.get("crashed", [float("nan")] * 2)
    return c[0], c[1]


def fig_marginals(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(DIMS), figsize=(3.0 * len(DIMS), 3.4),
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (dim, label) in zip(np.atleast_1d(axes), DIMS.items()):
        _style(ax, grid_axis="y")
        for arm, color in zip(ARMS, [SERIES_BLUE, CRITICAL]):
            d = df[df.arm == arm]
            if d[dim].nunique() < 4:
                continue
            q = pd.qcut(d[dim], 4, duplicates="drop")
            rate = d.groupby(q, observed=True).failed.mean()
            centres = [iv.mid for iv in rate.index]
            ax.plot(centres, 100 * rate.to_numpy(), marker="o", ms=4,
                    color=color, linewidth=1.5, label=ARM_LABEL[arm], zorder=2)
        ax.set_xlabel(label)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("failure rate (%)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle("Per-dimension failure rate", color=INK, fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_paired(df: pd.DataFrame, out: Path) -> pd.Series:
    piv = df.pivot_table(index="scenario_id", columns="arm", values="margin")
    piv = piv.dropna(subset=[a for a in ARMS if a in piv.columns])
    diff = (piv["neural"] - piv["classical"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="y")
    colors = [CRITICAL if d < 0 else SERIES_BLUE for d in diff]
    ax.bar(range(len(diff)), diff.to_numpy(), color=colors, alpha=0.75,
           width=1.0, zorder=2)
    ax.axhline(0.0, color=INK, linewidth=1.2, zorder=3)
    ax.set_xlabel("scenario (sorted)")
    ax.set_ylabel("margin: neural - classical")
    _title(ax, "Paired comparison on identical scenarios",
           f"neural worse in {100 * (diff < 0).mean():.0f}% of "
           f"{len(diff)} paired scenarios")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="analysis_px4/report")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, df = load(Path(args.results))
    fig_boundary(df, out / "boundary.png")
    med_cls, med_nn = fig_severity(df, out / "severity.png")
    fig_marginals(df, out / "marginals.png")
    diff = fig_paired(df, out / "paired.png")

    fail = df.groupby("arm").failed.mean()
    crash = df[df.status == "crashed"].groupby("arm").size()
    lines = [
        "# PX4 neural controller vs classical stack — adversarial campaign",
        "",
        f"Runs: {len(raw)} attempted, {len(df)} scored "
        f"({100 * len(df) / max(len(raw), 1):.0f}%). "
        f"Settled at setpoint before scoring: "
        f"{int(df.get('settled', pd.Series(dtype=bool)).sum())}.",
        "",
        "## Headline",
        "",
        "| metric | classical | neural |",
        "|---|---|---|",
        f"| failure rate | {100 * fail.get('classical', float('nan')):.1f}% "
        f"| {100 * fail.get('neural', float('nan')):.1f}% |",
        f"| crashes | {int(crash.get('classical', 0))} "
        f"| {int(crash.get('neural', 0))} |",
        f"| median lateral drift, crash scenarios | {med_cls:.2f} m "
        f"| {med_nn:.2f} m |",
        "",
        f"Paired on identical scenarios, the neural controller scores a lower "
        f"margin in {100 * (diff < 0).mean():.0f}% of {len(diff)} scenarios "
        f"(mean difference {diff.mean():+.3f}).",
        "",
        "## Reading it",
        "",
        "Both controllers lose altitude under a large enough mass step, so the "
        "pass/fail boundary is similar. The difference is in HOW they fail: "
        f"in crash scenarios the network's lateral drift median is "
        f"{med_nn / med_cls:.1f}x the classical stack's "
        "— it loses position control as well as altitude, where PID holds "
        "lateral position while descending.",
        "",
        "## Caveats",
        "",
        "- SIH (simple internal hover model), not Gazebo: aerodynamics are "
        "coarse and PX4's `failure motor off` injection is NOT honored by this "
        "backend (verified inert), so motor_fail rows are a null lever kept "
        "visible on purpose.",
        "- The policy shipped in mainline is trained for an X500 V2 frame; the "
        "SIH quad is not that airframe, so some degradation is expected and "
        "the classical arm is the control for exactly that reason.",
        "- Hover/position-hold only. No trajectory tracking, no wind field.",
        "- Oracle clamps any crash to margin -1.0, so severity beyond the "
        "crash threshold lives in `max_dev_m`, not in the margin.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/ (boundary, severity, marginals, paired, REPORT.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
