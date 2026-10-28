# Baseline preserved before thermal-supervision design

Date: 2026-10-29 (Asia/Shanghai).

- Branch at start: `codex/n1b-real-geometry`.
- Existing project HEAD: `21454d8` (Complete three-seed Coffee confirmation and checkpoint audits).
- No modified tracked source files were present at the start of this task.
- The previously untracked `reports/advisor_pre_2026-10-27.md` is included in this snapshot as a historical report. Its experiment status predates the completed N1b v4 confirmation; the latest evidence is in `experiments/n1b_v4/README.md` and its recorded results.
- The FasterGS4D and ICGui repositories remain pinned by their existing Git submodule references.
- Untracked third-party RoMa checkout: `experiments/f1m/match_dependencies/RoMa`, upstream `https://github.com/Parskatt/RoMa.git`, commit `77f8d68803526dcddfd9b7a46bc76125bdc25f15`. Its working tree was clean. This downloaded dependency is kept locally and recorded by revision, rather than vendored as an accidental embedded repository.
- Datasets, environments, caches and trained checkpoints remain local under the existing repository policy. This snapshot is a source-code backup, not a backup of model weights or raw data.
- No new training is authorized or started by this design task.

The new thermal design must compare against both native FasterGS4D and the existing N1b v4 acceleration variant. The previously observed Coffee gain is not evidence for thermal supervision or low-light/deblurring performance.
