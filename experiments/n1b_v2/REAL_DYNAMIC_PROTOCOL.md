# Measured-shutter dynamic scene validation

## Selection and provenance

The selected dataset is the real [DSEC](https://github.com/uzh-rpg/DSEC) RGB stereo capture. Its [official format documentation](https://dsec.ifi.uzh.ch/data-format/) provides per-frame **exposure start and end in microseconds**, rectified RGB images and intrinsic/extrinsic camera calibration. The [paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf) describes hardware-triggered global-shutter RGB cameras at 20 Hz. No exposure is inferred from the frame rate.

Download real PNG members with HTTP ranges from the authors' original archives, check ZIP CRC32, and save SHA256 per file. Download both exposure tables, unified timestamps and calibration. The source URLs, archive sizes/ETags and every downloaded image are in `dataset/dsec_dynamic/<sequence>/download_manifest.json`. There are no generated or averaged blurry training images.

Three clips, fixed before running confirmation seeds:

- `zurich_city_04_c`, frames 120–143: day, passing tram and leading vehicle.
- `zurich_city_04_c`, frames 260–283: day, traffic approaching an intersection.
- `zurich_city_09_a`, frames 120–143: night, exposure stress test with traffic. Both RGB exposures are measured 14.996 ms; day exposures are substantially shorter and are read per frame.

HDR-GoPro was also investigated and a training image/calibration sample downloaded. The paper gives absolute per-scene shutter settings (e.g. tumbler 1/120, 1/480, 1/1920 s), but the inspected public package did not establish physical frame intervals and image-crop/calibration consistency sufficiently for this exposure-integration experiment. It is not silently assigned an assumed frame rate. Fraunhofer's real motion-blur dataset requires an email access request; no message was sent.

## Time, coordinates and held-out data

Use exact left/right exposure starts and ends. Subtract a local microsecond origin in float64, convert to seconds, then divide by the clip span. Record both physical and normalized times. No large absolute timestamp is cast directly to float32.

The calibrated stereo rig is the coordinate gauge. Left rectified camera is identity; right is `R_rect2 @ T_21 @ inverse(R_rect1)`. The 4D scene represents both independent object motion and ego-induced motion relative to this rig. This is a legitimate time-varying coordinate choice, but there is **no recovered world trajectory or physical object-velocity claim**. It does not approximate a changing world camera by fixing it while holding the world scene static.

Resize 1440×1080 to 360×270 with Lanczos and scale all four intrinsic values accordingly. No cropping. For each clip, with local frame indices 0–23, hold out indices 2,6,10,14,18,22 from **both cameras**. The other 18 stereo pairs are training. Thus each clip has 36 train observations and 12 held-out observations. This evaluates new times at calibrated views, not an entirely unseen camera.

Stereo SGBM initialization uses only the 18 training stereo pairs. Test timestamps and pixels never enter depth initialization or training. Use 32,768 Gaussian samples from valid stereo depth; no test-depth or pretrained scene checkpoint. Shared 1,200-step warmup (Gauss8, repaired position coordinates, frozen temporal widths) produces one identical starting checkpoint for all arms.

## Frozen arms and budgets

Following a two-step smoke test and day/120/seed-0 pilot, keep hyperparameters fixed for all three clips × seeds 0,1,2, including pilot. Do not select checkpoints or hyperparameters on held-out metrics.

Each branch performs **800 steps with eight native full-frame renders/step**, identical deterministic training-observation order and nuisance parameter groups:

| Arm | Temporal quadrature | XYZ coordinates / LR | log-time-width LR |
|---|---|---|---|
| original | archived N1b-v2 balanced partition | raw world XYZ / .01 | .01 |
| modified | same partition | frozen conditional-covariance-normalized XYZ / .01 | .05 |
| Gauss | global Gauss8 | same repaired coordinates / .01 | .05 |

All branches train SH0 (.01), SH-rest (.0005), opacity logits (.01), spatial log scales (.003), temporal means (.0005), left and right 4D isoclinic rotations (.0002). All XYZ coordinates remain trainable. Full 4D covariance is enabled; these are not identity-rotation static-scene models. No densification or pruning. The added shared appearance/motion parameter groups and real-data adapter are explicitly **not a verbatim execution of the archived eight-Gaussian synthetic training script**.

Primary comparison is equal updates/data/native render budget, not equal wall time. Record wall time without a speed claim; desktop GPU load is not isolated. Gauss uses the repaired optimizer so optimizer improvements cannot be attributed to temporal partitioning. Identical-schedule fractions are logged.

## Evaluation and limits

Evaluate every branch using **the same Gauss32 exposure integral** over each held-out target's actual measured shutter, with fixed rectangular weighting. This measures reconstruction of real recorded dynamic frames, not just training fit. Accumulate a Gauss16-vs32 convergence diagnostic. Compute float RGB PSNR, SSIM, LPIPS-VGG, and fixed-ROI PSNR; full-frame measures can be dominated by background. Rectangular ROIs (center truck area plus left foreground tram area) are specified in code before training; they are **not ground-truth dynamic segmentation masks**, and night ROIs are only a common spatial diagnostic.

DSEC does not supply paired instantaneous sharp RGB ground truth for these clips. Results can establish held-out dynamic **finite-exposure frame reconstruction** gains, but cannot prove deblurring or instantaneous sharp-image recovery. Camera radiometric response and within-exposure hardware weighting are not fully calibrated in this experiment; all branches use the same RGB-domain uniform-shutter approximation. Auto-exposure and gain are part of the real observations, not replaced by assumed shutter values.

Two day clips share a capture and all clips are short. Treat them as three clip/seed comparisons, not 108 independent scenes. No comparison with DSEC's optical-flow/disparity leaderboard or full production-scale 4DGS is implied.

## Reproduction

```powershell
$env:PATH='E:\4D_Gaussian\.conda\nerficg\Library\bin;'+$env:PATH
& .conda/nerficg/python.exe -m experiments.n1b_v2.download_dsec --sequence zurich_city_04_c --start 120 --count 24
& .conda/nerficg/python.exe -m experiments.n1b_v2.download_dsec --sequence zurich_city_04_c --start 260 --count 24
& .conda/nerficg/python.exe -m experiments.n1b_v2.download_dsec --sequence zurich_city_09_a --start 120 --count 24
& .conda/nerficg/python.exe -m experiments.n1b_v2.real_dynamic --start 120 --seed 0 --out output/new_dynamic_run
```

Repeat the last command with day start 260 and night `--root dataset/dsec_dynamic/zurich_city_09_a`, for all three seeds. Output directories must not already exist. Models, predictions, calibration/time manifest and original metrics remain in each output directory. Compact results and an independent audit are retained in `experiments/n1b_v2/results/real_dynamic`.
