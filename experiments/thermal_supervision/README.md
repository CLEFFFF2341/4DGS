# CTG-GS design primitives — not a runnable reconstruction pipeline

The user requested an algorithm design with independent review and **no training**.
This directory provides small, tested PyTorch building blocks for that design.
It does not import the native renderer, create a Trainer, load images, optimize a
scene, start a GPU job, or create a model checkpoint.

- [Algorithm and implementation plan](../../research/thermal_supervision_20261029/PROPOSAL_zh.md)
- [Related work and novelty boundaries](../../research/thermal_supervision_20261029/LITERATURE.md)
- [Data requirements and evaluation protocol](../../research/thermal_supervision_20261029/EVALUATION_zh.md)
- [Independent review](../../research/thermal_supervision_20261029/REVIEW.md)

`core.py` provides exposure averaging in the linear sensor domain, a robust
noise-scaled observation loss, scalar thermal SH0 packing, explicit gradient
routing, a functional bounded proposal, and a fail-closed acceptance decision.
The decision consumes pre-rendered geometry/appearance trial observations.
Source-only candidate search, calibration, overlap masks, real rasterization,
transaction commit, native cleanup and checkpoint support are **not implemented**.
Passing tests cannot be interpreted as end-to-end implementation feasibility or
evidence of higher PSNR.

CPU checks in the existing environment:

```powershell
$env:PATH = 'E:\4D_Gaussian\.conda\nerficg\Library\bin;' + $env:PATH
$env:CUDA_VISIBLE_DEVICES = ''
& E:/4D_Gaussian/.conda/nerficg/python.exe -m unittest experiments.thermal_supervision.test_core -v
```

Recorded result on 2026-10-29: **22 tests passed**. No training, optimizer step, dataset-driven
inference or GPU rendering was executed. Checks cover gradients, exposure
normalization, unsafe evidence rejection, test-split exclusion, cumulative RGB
damage limits, and overflow cases identified by the independent reviewer.

`Frame.calibrated=False` is the default. A caller must supply actually audited
calibration and synchronized observations; changing this Boolean is not a
calibration procedure. Noise and mask inputs are fixed throughout a transaction.
For scene motion, witnesses require distinct viewpoints at the same physical
time within a documented synchronization tolerance.
