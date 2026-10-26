# Execution notes

- Prior results/code committed as `4ba6e41`; full-native protocol and runner
  committed before test evaluation as `c8aefa5`.
- Full native Trainer and MortonEncoding imported successfully on this run.
  No extension rebuild, binary substitution, or security-setting change.
- Initial zero-field smoke checks exposed nondeterminism in repeated native
  rendering of the overlapping stereo point cloud. The strict image-equality
  assertion failed before training. A repeated-native control showed comparable
  differences. The audit now checks exact coordinate equality and bounds image
  and gradient discrepancies by this measured native control. The candidate
  algorithm was not changed. The successful 6,001-step smoke included native
  density/opacity/Morton callbacks, and computed no test metric.
- Full DSEC city_05_a seed 3 native run was interrupted for resource pressure,
  not scored as a quality loss. It reached 6,001 steps / 1,815,303 Gaussians in
  16.08 seconds, but reached 7,001 / 2,508,023 only at 210.80 seconds, with
  observed dedicated GPU memory near 24 GiB. The last progress report arrived
  just before interruption. This was severe slowdown, not proof of a deadlock
  or a CUDA OOM exception. The remaining DSEC full-native suite is deferred.
  No density threshold or capacity cap was changed to make it fit.
- Priority moved to the predeclared full coffee_martini pair. An incomplete
  DSEC experiment cannot establish or refute a native-baseline quality gain.
- Subsequent investigation found a launcher-order issue: this harness loaded
  the Dataset before creating the Trainer, whereas official `train.py` creates
  the Trainer first. On Windows this let existing CUDA allocations mask the
  unsupported expandable-segment warning; the native feature probe then skipped
  cache clearing. The first coffee attempt was also stopped before evaluation.
  The runner now follows the official construction order, including creating
  the candidate's field only after the parent Trainer constructor. No training
  algorithm or threshold was changed. Audit rows record allocator flags/bytes.
- The first restart exited with code -1073740791 near a Windows nvlddmkm event
  153. A fresh CUDA matrix-multiplication check succeeded. The next clean DSEC
  restart (`native03_`) used `requires_empty_cache=True` and reached 9,001 steps /
  4,033,545 Gaussians in 45.67 seconds with about 5.5 GB reserved. This restores
  feasibility of the original full DSEC plan; earlier resource-pressure notes
  describe the interrupted attempts, not a native algorithm limitation.
