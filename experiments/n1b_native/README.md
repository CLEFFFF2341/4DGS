# N1b successor versus the complete native FasterGS4D trainer

This experiment transfers the frozen 96-parameter temporal affine field from
`experiments/n1b_v3` to the complete native training pipeline. It is distinct
from the earlier fixed-capacity, shared-warmup experiments.

Read `PROTOCOL.md` for the predeclared comparison, `TRANSFER_GATE.md` for the
coffee replication criterion, and `RUN_LOG.md` for infrastructure diagnostics.
Completed metrics and conclusions are archived in `results/` and `RESULTS.md`.

## Implementation boundary

- `native.py`: inherited native training/density/opacity/Morton/SH callbacks;
  the candidate changes the rendered spatial means and adds a separate native
  FusedAdam for 96 global parameters. Audits check identical initialization,
  zero-field coordinates, native numerical repeatability and callback execution.
- `data.py`: the existing DSEC split/stereo point-cloud adapter, and the same
  verified lossless coffee cache used by the original deployment.
- `run.py`: one independent process per arm, native model/Trainer construction
  **before** Dataset loading, full training, checkpoint integrity and evaluation.
- `suite.py`: sequential GPU queue; refuses to overwrite unfinished runs.
- `assess.py`: pairs completed 30k-step runs and checks hashes/callback counts.
- `recover_eval.py`: verifies a completed checkpoint and restores missing
  evaluation without training again.
- `finish.py`: one-shot continuation after the active queue exits. It completes
  missing jobs, applies the predeclared coffee replication gate, audits all
  pairs, writes the final report and optionally commits/pushes this experiment.
  Incomplete training is preserved for inspection, never overwritten.

No source file in the native FasterGS4D submodule is changed. Its revision is
`3dbbb985c5e0f29825d0960563f3c39f29eb3987`. Native source hashes are recorded for
each run. Logging and checkpoint/export settings differ from the production
entry point; the model updates and density schedule are inherited unchanged.

## Reproduction (PowerShell, from repository root)

```powershell
$env:PATH = "$PWD\.conda\nerficg\Library\bin;" + $env:PATH
.\.conda\nerficg\python.exe -m experiments.n1b_native.suite --dataset dsec --prefix replication_
.\.conda\nerficg\python.exe -m experiments.n1b_native.suite --dataset coffee --prefix replication_
.\.conda\nerficg\python.exe -m experiments.n1b_native.assess
```

The active continuation uses prefix `native03_`. Its final status will be written
to `output/n1b_native/completion.json`, or a failure to `completion_error.json`.
The one-shot continuation can also be invoked after the queue is stopped:

```powershell
.\.conda\nerficg\python.exe -m experiments.n1b_native.finish --publish
```

Do not start a second continuation while one is already running. The `--wait-pid`
option waits for the specifically identified active queue process before doing
any work; it does not interrupt that process or alter GPU scheduling.

The official local datasets, CUDA extensions and VGG metric weights must already
be installed. See `deployment/` and the earlier N1b data manifests. Models and
large local data remain under ignored `output/`; scalar metrics, per-view records,
configuration/provenance and selected comparisons are committed.

## Interpretation

The primary comparison is native midpoint inference with official 8-bit image
quantization and VGG LPIPS. DSEC additionally reports common Gauss32 integration.
Same iterations/render count does not guarantee identical runtime or Gaussian
count: density control remains active. DSEC uses a camera-rig coordinate adapter
and a narrow stereo camera extent; it is not an official DSEC benchmark. Coffee
uses the original multi-camera loader, split and full resolution. A result from
one scene/seed alone must not be described as broad or stable superiority.
