"""Side-by-side crash video: one scenario, two controllers.

    python analysis_px4/render_pair.py --classical <ulg> --neural <ulg> \
        --out docs/assets/neural-pair.mp4 [--still frame.png]

Reads PX4 ulogs, syncs both flights on their own injection timestamp (the
SIH_MASS parameter-change record), and renders a scope-styled animation in the
site's palette: top-down position trail against the 3 m limit circle, altitude
strip against the crash floor, live readouts, verdict stamps.

--still renders a single end-state frame (fast) for layout iteration and for
the <video> poster image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from pyulog import ULog  # noqa: E402

# site palette (docs/index.html :root)
INK0, INK1 = "#0a0d11", "#0e1319"
HAIR, HAIR_SOFT = (0.66, 0.75, 0.85, 0.16), (0.66, 0.75, 0.85, 0.09)
TEXT, TEXT2, TEXT3 = "#e8ecf2", "#a4aebc", "#6b7684"
SAFE, SAFE_BRIGHT = "#6f92c8", "#8fb0e0"
FAIL, FAIL_BRIGHT = "#d64541", "#ef6b60"
CAUTION, SCOPE = "#d9a23f", "#84d69c"

MONO = {"family": "Menlo", "weight": "regular"}
T_PRE, T_POST = 6.0, 26.0     # seconds shown around the injection
FPS, PLAY_RATE = 20, 1.5      # 32 s of sim in ~21 s of video
POS_LIM, CRASH_ALT, SETPOINT = 3.0, 0.3, 5.0


def load(path: str):
    u = ULog(path, ["vehicle_local_position"])
    d = next(x for x in u.data_list if x.name == "vehicle_local_position")
    t = d.data["timestamp"] / 1e6
    inj = next(ts / 1e6 for ts, n, _ in u.changed_parameters
               if n == "SIH_MASS")
    m = (t >= inj - T_PRE) & (t <= inj + T_POST)
    tt = t[m] - inj
    x, y, alt = d.data["x"][m], d.data["y"][m], -d.data["z"][m]
    # hold-point = mean position over the settled pre-event window
    pre = tt < -1.0
    x, y = x - x[pre].mean(), y - y[pre].mean()
    return {"t": tt, "x": x, "y": y, "alt": alt,
            "drift": np.hypot(x, y)}


def style_scope(ax, label, color):
    ax.set_facecolor(INK1)
    ax.set_aspect("equal")
    lim = 3.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for r, ls, c, lw in [(1.0, ":", HAIR, 0.8), (2.0, ":", HAIR, 0.8),
                         (POS_LIM, "--", FAIL, 1.2)]:
        ax.add_patch(Circle((0, 0), r, fill=False, ls=ls, ec=c, lw=lw))
    ax.plot([-0.28, 0.28], [0, 0], color=HAIR, lw=0.9)
    ax.plot([0, 0], [-0.28, 0.28], color=HAIR, lw=0.9)
    ax.text(0.04, POS_LIM + 0.12, "3 m limit", color=FAIL_BRIGHT, fontsize=7.5,
            **{"fontfamily": "Menlo"})
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(HAIR); s.set_linewidth(0.8)
    ax.set_title(label, color=color, fontsize=11, fontweight="bold",
                 fontfamily="Menlo", pad=8)


def style_strip(ax):
    ax.set_facecolor(INK1)
    ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(0, 6.4)
    ax.axhline(SETPOINT, color=HAIR, lw=0.9, ls=":")
    ax.axhline(CRASH_ALT, color=FAIL, lw=1.0, ls="--")
    ax.axvline(0, color=CAUTION, lw=1.0)
    ax.text(0.4, 0.62, "plant step", color=CAUTION, fontsize=7.5,
            fontfamily="Menlo")
    ax.text(-T_PRE + 0.4, SETPOINT + 0.22, "setpoint 5 m", color=TEXT3,
            fontsize=7.5, fontfamily="Menlo")
    ax.text(-T_PRE + 0.4, CRASH_ALT + 0.18, "ground", color=FAIL_BRIGHT,
            fontsize=7.5, fontfamily="Menlo")
    ax.tick_params(colors=TEXT3, labelsize=7.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily("Menlo")
    for s in ax.spines.values():
        s.set_color(HAIR); s.set_linewidth(0.8)
    ax.set_ylabel("alt m", color=TEXT2, fontsize=8, fontfamily="Menlo")
    ax.set_xlabel("seconds from plant step", color=TEXT2, fontsize=8,
                  fontfamily="Menlo")


def build_figure(runs):
    fig = plt.figure(figsize=(11.6, 6.6), dpi=110)
    fig.patch.set_facecolor(INK0)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0],
                          left=0.06, right=0.97, top=0.86, bottom=0.09,
                          hspace=0.34, wspace=0.16)
    fig.text(0.06, 0.955, "ONE SCENARIO. TWO CONTROLLERS.", color=TEXT,
             fontsize=15, fontweight="bold", fontfamily="Menlo")
    fig.text(0.06, 0.915,
             "mass ×1.83 + thrust ceiling ×1.19 stepped mid-hover · "
             "SIH quadrotor · PX4 mainline 544bccbc · scenario 115 of 400",
             color=TEXT3, fontsize=8.5, fontfamily="Menlo")

    arts = {}
    for col, (arm, color, bright) in enumerate(
            [("classical", SAFE, SAFE_BRIGHT), ("neural", FAIL, FAIL_BRIGHT)]):
        ax = fig.add_subplot(gs[0, col])
        label = ("CLASSICAL (PID)" if arm == "classical"
                 else "NEURAL (mc_nn_control)")
        style_scope(ax, label, bright)
        trail, = ax.plot([], [], color=color, lw=1.4, alpha=0.85, zorder=3)
        head, = ax.plot([], [], marker="o", ms=6, color=bright, zorder=4)
        readout = ax.text(0.03, 0.03, "", transform=ax.transAxes, color=TEXT2,
                          fontsize=8.5, fontfamily="Menlo", va="bottom")
        stamp = ax.text(0.5, 0.24, "", transform=ax.transAxes, ha="center",
                        color=bright, fontsize=17, fontweight="bold",
                        fontfamily="Menlo", alpha=0.0, zorder=6)
        axs = fig.add_subplot(gs[1, col])
        style_strip(axs)
        altline, = axs.plot([], [], color=color, lw=1.4)
        arts[arm] = dict(trail=trail, head=head, readout=readout, stamp=stamp,
                         altline=altline, data=runs[arm])
    clock = fig.text(0.97, 0.955, "", color=SCOPE, fontsize=11,
                     fontfamily="Menlo", ha="right")
    return fig, arts, clock


def draw_at(t_now, arts, clock):
    clock.set_text(f"T{t_now:+06.1f}s")
    for arm, a in arts.items():
        d = a["data"]
        m = d["t"] <= t_now
        if not m.any():
            continue
        a["trail"].set_data(d["y"][m], d["x"][m])       # NED: y=east on x-axis
        a["head"].set_data([d["y"][m][-1]], [d["x"][m][-1]])
        i = int(np.flatnonzero(m)[-1])
        a["altline"].set_data(d["t"][m], d["alt"][m])
        drift, alt = d["drift"][i], d["alt"][i]
        a["readout"].set_text(f"drift {drift:4.2f} m   alt {alt:4.2f} m")
        crashed = (d["alt"][m] < CRASH_ALT).any() and t_now > 1
        ended = t_now >= d["t"][-1] - 0.5
        if crashed:
            a["stamp"].set_text("CRASHED")
            a["stamp"].set_alpha(min(1.0, a["stamp"].get_alpha() + 0.12))
        elif ended:
            a["stamp"].set_text("HELD")
            a["stamp"].set_alpha(min(1.0, a["stamp"].get_alpha() + 0.12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", required=True)
    ap.add_argument("--neural", required=True)
    ap.add_argument("--out", default="docs/assets/neural-pair.mp4")
    ap.add_argument("--still", default=None,
                    help="render one end-state PNG instead of the video")
    args = ap.parse_args()

    runs = {"classical": load(args.classical), "neural": load(args.neural)}
    fig, arts, clock = build_figure(runs)

    if args.still:
        draw_at(T_POST, arts, clock)
        for a in arts.values():
            a["stamp"].set_alpha(1.0)
        fig.savefig(args.still, facecolor=INK0)
        print(f"wrote {args.still}")
        return 0

    n_frames = int((T_PRE + T_POST) / PLAY_RATE * FPS)
    writer = FFMpegWriter(fps=FPS, codec="h264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out), dpi=110):
        for f in range(n_frames + FPS * 2):        # +2 s freeze on verdict
            t_now = min(-T_PRE + f * PLAY_RATE / FPS, T_POST)
            draw_at(t_now, arts, clock)
            writer.grab_frame(facecolor=INK0)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{(n_frames + FPS * 2) / FPS:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
