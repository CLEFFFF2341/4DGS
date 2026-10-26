# Real measured-exposure validation

## Data and scope

- Authors' real [Ev-DeblurNeRF-CDAVIS](https://github.com/uzh-rpg/EvDeblurNeRF), all five scenes; not its synthetic Blender dataset.
- [Paper](https://rpg.ifi.uzh.ch/docs/CVPR24_Cannici.pdf): Color-DAVIS346, 100 ms exposure, encoder trajectory, separately captured stationary sharp references.
- Archive: 195,638,862 bytes; SHA256 `1117dd16caef0cc4e05db830119edf00860177bc7415f6778cc2bfb65e7c200a`.
- Local root: `dataset/evdeblurnerf_real/ev-deblurnerf_cdavis`.
- Recorded training shutter durations: 99.998–100.000 ms. Use each actual start/end in microseconds, subtract a local origin in float64, then convert to seconds. Do not infer exposure from frame rate.
- Official final five images per scene are sharp tests. Their zero-duration/zero timestamps are sentinels, **not measured zero-second shutters**. Training has 18/11/15/17/18 images respectively; all images 346×260.
- These are static scenes with moving cameras. Known positions are linearly interpolated throughout exposure; LLFF axes become native right/down/forward. Rotation is checked to be identity. At trajectory edges only, linear extrapolation is used and its duration logged per scene.
- Test images were captured separately and have no usable temporal timestamps. Query the model at the time corresponding to their slider position, recorded in each manifest. This convention is meaningful for this static-scene test and does not establish dynamic reconstruction quality.
- Integrate image intensities in the supplied RGB domain, consistently for all methods. There is no calibrated camera-response curve or measured continuous shutter weighting; a rectangular shutter within the recorded bounds is a model assumption, not independently measured hardware response.

## Frozen comparison

After a two-step smoke test and the first full batteries/seed-0 run, keep all hyperparameters unchanged and run **all five scenes × seeds 0, 1, 2**. Batteries/seed-0 is the pilot; other runs are confirmation, with no test-based tuning. Each run:

1. 8,192 Gaussians initialized by backprojecting **blurry training pixels only**, random depth 0.4–4 in dataset world coordinates. No external pretrained model, EDI deblurred image, events, test image, or test-derived point cloud.
2. Shared 1,200-step warmup: Gauss8, covariance-normalized XYZ Adam, time widths frozen. Save one identical checkpoint for all three branches.
3. Exactly 600 refinement steps, same training-view order, eight full-frame native renders per step. No densification or pruning. Three branches:
   - `original`: archived N1b-v2 balanced support partition; raw XYZ learning rate .01, log temporal width .01.
   - `modified`: same partition; frozen conditional-covariance-normalized XYZ learning rate .01, log temporal width .05, as in the preceding repair.
   - `gauss`: global Gauss8 with **the same improved optimizer** as modified.
4. All branches additionally optimize SH0 (.01), SH-rest (.0005), opacity logits (.01), and log spatial scales (.005), identically. Rotations and temporal centers fixed. All XYZ coordinates trainable. The common warmup and these appearance parameters are a real-data adaptation: these runs are **not** an unmodified execution of the archived eight-Gaussian synthetic harness.
5. Evaluate sharp held-out views only after all branches finish; report mean per-image PSNR and torchmetrics SSIM on clamped float RGB. Save predictions and checkpoints. Training error uses Gauss16; Gauss8-vs16 image MSE diagnoses integration error.

Primary fairness is equal updates, data, initialization, native render budget and nuisance parameters; wall times are recorded, **not equalized**. Some other desktop GPU load is possible. Timing is secondary, not a speedup claim. Gauss receives the optimizer repair so improvements over a poor raw-coordinate optimizer cannot be attributed to the partition rule.

## Implementation checks

`real_cdavis.schedule` accelerates interval discovery using sorted endpoints; it retains the archived fallback and balanced allocation. Compare against `partition.nodes` on random scales, rotations and supports; original endpoint subtraction precision is preserved. The full support/global-Gauss condition is logged at every update. No test selection, checkpoint selection, or early stopping.

## Reproduction (PowerShell)

```powershell
$env:PATH = 'E:\4D_Gaussian\.conda\nerficg\Library\bin;' + $env:PATH
& .conda/nerficg/python.exe -m experiments.n1b_v2.download_cdavis
& .conda/nerficg/python.exe -m unittest experiments.n1b_v2.test_real_cdavis
& .conda/nerficg/python.exe -m experiments.n1b_v2.real_cdavis --scene blurbatteries --seed 0 --out output/new_cdavis_run
```

Output directories must not already exist. Defaults reproduce 8,192 points, 1,200 warmup steps, 600 refinement steps, Gauss/partition order 8. Repeat with all scene names and seeds. Raw data and large checkpoints stay outside Git; compact metrics and source remain in the experiment directory.
