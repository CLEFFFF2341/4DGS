# N1b v4: revise using internal validation, then freeze for native confirmation

The v3 global affine field did not pass the full coffee native transfer gate:
PSNR -0.075604 dB, SSIM +0.000709, VGG LPIPS -0.000734 at seed 0. It must not be
described as a stable improvement. This is a new development experiment.

## Development data and budget (fixed before these runs)

- Native coffee loader, official area resize to scale 0.125; every tenth frame.
- Remove cam08 and cam16 from gradient supervision. Cam08 is primary development
  validation; cam16 is reserved for a second check after choosing a configuration.
- The native supplied point cloud is shared unchanged. It was reconstructed
  before this internal split, so this is not a claim of camera-independent
  initialization. No existing full-data trained checkpoint is used for selection.
- Official test cam00 images are not loaded, rendered or scored during selection.
  Prior cam00 results motivated the hypotheses, so do not call that dataset wholly
  unseen. Fresh seeds and frozen settings are needed for subsequent confirmation.
- Train an unmodified native prefix for 15,000 steps, batch 1. Save raw parameters,
  optimizer moments/steps, sampler cursor/indices, CPU/CUDA RNG, and native extent
  before final cleanup. Branches resume the same state to a total 30,000 steps.
- All original density, opacity, SH, Morton, loss and learning-rate settings stay
  unchanged. After the native density phase, optional residuals start at zero.
  Baseline also resumes the same prefix with its optimizer and sampling state.

## Hypotheses to separate

1. The global field may interfere with density development. Test the same field
   activated only after the shared native prefix.
2. A Gaussian's curved motion may need a local residual, not a scene-wide affine
   displacement. Test three zero-initialized acceleration coefficients per point:
   delta_mu = conditional_std_ref * a * clamp(q,-3,3)^2, where
   q=(t-current_tau)/marginal_temporal_std_ref. Position and velocity at tau are
   preserved; scale features are frozen from the shared prefix.
3. Native SH colors lack an explicit temporal term. Test three per-point SH0
   slope coefficients: delta_SH0 = b * clamp(q,-3,3). This uses time and Gaussian
   state, not camera identity or test-image information.
4. Test the combination, and modest residual-learning-rate alternatives if needed.

Initial arms: native; late global affine (3e-5); acceleration (0.01); temporal
color (0.001); acceleration+color (0.01/0.001). Native loss stays unchanged in
these arms. If loss tuning is tried later, include the same loss on a native
control; do not attribute an objective change to the representation.

Per-point residuals and their optimizer states must follow native pruning and
sorting. They are never active during densification. Validate this correspondence
and zero-residual rendering before the real comparisons. Report extra parameters,
capacity, training time and all negative results.

## Selection and confirmation

Use primary validation mean PSNR, with SSIM/LPIPS checked for regressions. Require
at least +0.10 dB and no mean SSIM/LPIPS regression before advancing a candidate.
Repeat at a second development seed and evaluate the reserved cam16 split only
after freezing the chosen arm. Failing candidates remain development results.

Then run full-resolution, all-training-camera, batch-4 native comparisons at
30,000 total steps with fresh seeds 1,2,3, common native 15k prefixes, and all
300 official test images. Report each pair and require positive PSNR at all
three seeds, mean >= +0.10 dB, and no overall SSIM/LPIPS regression for a claim
of repeatable improvement on coffee. Do not generalize that claim to other scenes
without further frozen validation. Never guarantee that the search will succeed.

## Round 2 (declared after round 1 primary results, before new metrics)

Native, affine, acceleration and color primary PSNR were 23.074614, 23.088057,
23.073717 and 23.066907 dB. None advances. The combination process failed during
startup with Windows access violation and an empty log, before creating an output
directory; retry it unchanged and retain the failure log.

Next arms use the same native seed-0 prefix: acceleration LR 0.05; color LR 0.005;
native + 2*MSE; native + 8*MSE; acceleration LR 0.01 + 2*MSE. MSE is mean RGB
squared error on the same rendered images. Original 0.8*L1 + 0.2*DSSIM remains.
These are late-stage training-objective experiments, not evidence of a new
representation by themselves. Compare acceleration+MSE against native+sameMSE
to separate any structural contribution. No official test or reserved camera
is evaluated in this round. The original advancement gate remains unchanged.

## First frozen structural candidate

After acceleration LR 0.05 completed, before the round-2 MSE arms completed,
freeze **kind=accel, accel_lr=0.05, no color, no MSE**, activated after step 15,000.
Seed-0 primary: PSNR 23.22826570 (+0.15365124 dB), SSIM 0.86067807
(+0.00254131), LPIPS 0.10891085 (-0.00251786). This passes the development gate.
Run a fresh shared native seed-1 prefix and native/candidate continuations;
do not adjust the frozen candidate using seed-1 or reserved cam16 outcomes.
Finish the already declared other round-2 arms for transparent evidence, without
silently replacing this candidate based on confirmation results.

Seed-1 primary replication passed without changing settings: +0.152331 dB,
SSIM +0.005085, LPIPS -0.008403. Before reading any candidate cam16 metric,
the reserved-view decision is: positive PSNR at both seeds and non-regressing
mean SSIM/LPIPS across the two pairs. Failure blocks full confirmation for this
frozen candidate; do not tune to the reserved camera.
