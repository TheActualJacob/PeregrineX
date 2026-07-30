# Kaggle side of the AviationSim workflow

The split: **Kaggle owns GPUs and Linux builds; the Mac owns SITL evidence
generation.** Train learned params here, bring the exported model home, and
certify it with `sim/harness` (430–450 runs/hr on the M4 Air). Kaggle quotas:
~30 h/week GPU (T4×2 / P100), 12 h per session, ~4 vCPUs — a Kaggle SITL farm
runs at roughly a third of the Mac's throughput, so use Kaggle for what the
Mac can't do, not for runs.

## Pieces

| Piece | Where | What |
|---|---|---|
| `jacobruan1/aviationsim-week1-evidence-v3` | Kaggle dataset (private) | 2,000-run baseline + 500-run shakeout parquets, all 2,543 trajectory sidecars concatenated into `trajectories.parquet` (run_id column fixed), per-run metadata (`run_meta.jsonl`), harness/oracle code snapshot (`code.zip`), pinned `ARDUPILOT_COMMIT` |
| `train_abort_monitor/` | GPU script kernel | Trains the abort/go-around monitor (GRU) on the sidecars; outputs `abort_monitor.pt`, `abort_monitor.onnx`, `eval.json` |
| `sitl_farm_bootstrap/` | CPU script kernel (needs internet) | Builds ArduPlane SITL at the pinned commit, smoke-tests a heartbeat, outputs `sitl_plane_build.tar.gz` for reuse |

## Gotchas learned setting this up

- In the raw sidecars (on the Mac), the `run_id` **column** holds the dataflash
  ordinal (`00000001`); the true run_id is in the **filename**. The dataset's
  `trajectories.parquet` already has this fixed.
- Datasets containing a many-file `.tar.gz` got **stuck in processing forever**
  (slugs `...-evidence` and `...-evidence-v2` are zombies: 403 on every API call
  including delete). A one-file probe processed in 30 s; flat parquet/jsonl/zip
  files work. Don't upload tarballs.
- A `kaggle kernels push` immediately **queues a run** on Kaggle's servers.
- A freshly created dataset takes minutes to process; kernels pushed before
  it is ready silently lose the dataset attachment (`not valid dataset
  sources`) — re-push after `kaggle datasets status <slug>` says `ready`.
- Kernel `title` must slugify to the metadata `id`, or Kaggle renames the
  kernel URL out from under you.
- The bootstrap kernel needs internet (github clone), which requires a
  phone-verified Kaggle account.

## Commands

```bash
# re-push (and re-run) a kernel after editing
kaggle kernels push -p kaggle/train_abort_monitor

# watch a run
kaggle kernels status jacobruan1/aviationsim-train-abort-monitor
kaggle kernels status jacobruan1/aviationsim-ardupilot-sitl-build-smoke-test

# fetch outputs (trained model, eval.json) when complete
kaggle kernels output jacobruan1/aviationsim-train-abort-monitor -p /tmp/abort_monitor_out

# refresh the dataset after a new Mac campaign (bump from sim/):
#   re-run the staging script, then:
kaggle datasets version -p <staging-dir> -m "post-campaign refresh"
```

## PX4 neural track (the actual learned autopilot)

`px4_neural_build/` clones PX4 mainline and builds board config `px4_sitl_neural`
— SITL with `mc_nn_control` compiled in: a TFLite-Micro network whose learned
parameters ARE the position controller (15 state inputs → 4 motor commands),
not a learned decision bolted onto PID. Kernel outputs: `px4_build.tar.gz`
(attach to flight kernels via `kernel_sources`), `PX4_COMMIT`, `explore.txt`
recon (sihsim airframes, MC_NN params), and a first headless SIH boot log.
Flight kernels iterate on top of the cached build — no rebuild per attempt.

## Next steps queued behind this

1. **A/B on the Mac**: wrap `abort_monitor.onnx` in a pymavlink companion
   process (fires GUIDED go-around when P(fail) crosses the threshold chosen
   from `eval.json`), then re-run the 2,000-scenario baseline monitor-on vs
   monitor-off. The failure-rate delta, with the ±0.07 margin noise floor
   respected, is the certification-evidence demo.
2. **PX4 neural track**: PX4 mainline ships a TFLite neural position
   controller (`mc_nn_control`, board config `px4_sitl_neural`) that requires
   Ubuntu 24.04+ — a natural Kaggle kernel, zero Mac disk. Adversarial
   campaign against a mainline neural autopilot = flagship demo.
