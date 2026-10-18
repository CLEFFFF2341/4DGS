# FasterGS4D coffee_martini baseline

This repository preserves the working Windows experiment completed on 2026-10-18.
It includes the NeRFICG framework, a pinned official FasterGS4D submodule, the
experiment configuration, Windows preprocessing/training helpers, environment
manifests, measurements, and the paper reproduction audit.

```powershell
git clone --recurse-submodules git@github.com:CLEFFFF2341/4DGS.git
cd 4DGS
```

For an existing checkout, use `git submodule update --init --recursive`.
Both submodules use absolute upstream URLs so cloning this fork resolves them
independently of the fork's GitHub owner.

## Recorded baseline

| Setting or metric | Value |
|---|---|
| Framework upstream commit | e458ac6e41810ad6d9405687c1e698fadd28dac5 |
| Official FasterGS4D commit | 3dbbb985c5e0f29825d0960563f3c39f29eb3987 |
| Scene | Neural 3D Video coffee_martini |
| Train / test | 5100 training images / 300 held-out cam00 images |
| Resolution / frames | 1352 x 1014 / 300 frames per camera |
| Iterations / batch / SH degree | 30000 / 4 / 3 |
| Test PSNR / SSIM / LPIPS-VGG | 26.321117 / 0.893975 / 0.212569 |
| Training timer | 2669.079451 seconds |
| Peak CUDA allocated / reserved | 3.50 / 6.27 GiB |

**Workflow completion does not establish numerical reproduction of the paper.**
The paper's real-scene table covers three scenes, whereas this baseline includes
only coffee_martini. See [the audit](deployment/PAPER_REPRODUCTION_AUDIT.md).

## Reproducing and locating artifacts

Read [deployment instructions](deployment/README.md),
[measurements](deployment/RESULTS.md), and
[configuration](configs/fastergs4d_coffee_martini.yaml).
The helpers record the original machine paths; adjust those paths when running
on another machine. The scripts support the documented local deployment and are
not an unattended installer for every platform.

The conda environment, downloaded tools, input data, float32 caches, rendered
images, videos, and model checkpoints are excluded from Git. The completed local
run remains at
`E:\4D_Gaussian\output\FasterGS4D\fastergs4d_coffee_martini_2026-10-18-11-20-32`.
Checksums in `deployment/baseline_artifacts.json`, all 300 per-frame metrics in
`deployment/baseline_metrics_8bit.txt`, environment manifests, validation records, and selected successful
run logs are committed for provenance. This Git baseline alone does not contain
the trained model weights.

The framework's original README and license remain in place. The FasterGS4D
source is pinned to its official repository rather than copied into this fork.
