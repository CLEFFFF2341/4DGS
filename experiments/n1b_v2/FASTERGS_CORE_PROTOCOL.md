# Original FasterGS4D core follow-up

The full repository `FasterGS4DTrainer` was attempted first. Import fails when Windows Application Control blocks `MortonEncoding._C` with "An Application Control policy has blocked this file." No security setting was changed and no substitute for the blocked extension was installed. Native FasterGS rasterization, FusedAdam and FusedSSIM independently load and execute successfully.

Consequently this test is explicitly the **original FasterGS4D core, adapted to fixed-capacity refinement**, not a successful full default 30,000-step Trainer run. It directly imports the unmodified `src/Methods/FasterGS4D/Loss.py` source by filename; this avoids importing unrelated model/GUI components, not the blocked extension. Uses the installed original CUDA rasterizer and FusedAdam, whose source matches the repository. Original optimizer/loss scalar defaults are read from the original Trainer source AST.

Frozen data are the same three real DSEC clips, 360×270, train/test timestamps and seeds 0/1/2 as [the dynamic protocol](REAL_DYNAMIC_PROTOCOL.md). Every run loads **the exact existing shared_initial.pt** from its corresponding previous experiment. No new warmup, no test tuning, no changed initialization. All 32,768 Gaussians and all SH3 bases remain active.

## Original core settings

- One native render at each observation's measured mid-exposure timestamp for training and normal inference; no N1b or exposure quadrature in training.
- Unmodified FasterGS4DLoss: `.8 * L1 + .2 * (1 - fused_ssim)`.
- Original FusedAdam, beta defaults .9/.999, epsilon 1e-15.
- Original learning-rate groups: spatial/temporal means `.00016 * training_camera_extent`; SH0 .0025, SH-rest .000125, opacity .05, spatial/temporal log scales .005, left/right rotation .001.
- Camera extent uses the original formula `1.1 * max(norm(center - mean(center)))` over the same training cameras.
- Only spatial-mean learning rate decays exponentially to `.0000016 * extent` over 30,000 iterations, as in the source. Scheduler index starts at 1201 because the common initial checkpoint already has 1,200 warmup steps. Temporal-mean rate stays fixed.
- No covariance-normalized XYZ optimizer repair in this arm. Black background and all nine raw parameter tensors trainable.
- Densification, pruning, opacity resets, Morton ordering and SH ramp are excluded to preserve common capacity and initialization. This is a substantive scope limit, not claimed as the full default pipeline.

## Budgets and controls

Same deterministic training observation order as previous experiments; one observation per update. Save checkpoints at both 800 and 6,400 steps from one uninterrupted trajectory, then evaluate after all training finishes. Do not select between them based on test performance.

| Condition | Updates | Native training renders | Comparison |
|---|---:|---:|---|
| Previous modified N1b/Gauss8 | 800 | 6,400 | Existing results unchanged |
| Original FasterGS core, equal updates | 800 | 800 | Same number of updates, cheaper rendering |
| Original FasterGS core, equal renders | 6,400 | 6,400 | Same native render count, more updates |
| Matched midpoint control | 800 | 800 | Previous repaired optimizer + MSE + nuisance rates, with Gauss1 midpoint rendering |

The midpoint control isolates temporal integration from the original optimizer/loss changes. It is not labelled original FasterGS. The shared 1,200-step warmup costs 9,600 renders for every condition and is excluded identically from incremental budgets. Equal native renders do **not** imply equal total FLOPs or wall time: the original structural loss and optimizer differ.

## Evaluation

Report original single-time midpoint inference as the primary FasterGS output. Also render the same trained checkpoint through the common Gauss32 finite-exposure evaluator used previously, for a consistent observation model. Compute full-frame float RGB PSNR, SSIM, LPIPS-VGG and the unchanged fixed-rectangle ROI metric. Gauss32-vs64 is an independent integration check, not additional training.

No paired instantaneous sharp RGB truth exists in these clips. No physical deblurring or unseen-camera result is claimed. Full Trainer failure remains recorded rather than silently relabelling this limited adaptation as a full baseline reproduction.

## Reproduction

```powershell
$env:PATH='E:\4D_Gaussian\.conda\nerficg\Library\bin;'+$env:PATH
& .conda/nerficg/python.exe -m experiments.n1b_v2.real_fastergs --run output/dsec_c120_s0 --out output/new_fastergs_core_run
& .conda/nerficg/python.exe -m unittest experiments.n1b_v2.test_real_fastergs
```

Repeat for the nine prior run directories (`dsec_c120_s*`, `dsec_c260_s*`, `dsec_n120_s*`). New output directories must not already exist. Keep original raw run JSON, source/default fingerprints, midpoint and exposure predictions, and both model checkpoints.
