# Full native FasterGS4D comparison (2026-10-26)

Frozen before viewing any test metric from this experiment.

The hypothesis is that the already-frozen eight-knot affine center field from
N1b v3 transfers to the complete native trainer. The field has 96 zero-initialized
parameters and a fixed 3e-5 learning rate. No tuning on these test results.

Both arms inherit the unmodified native `training_iteration`, density control,
opacity resets, SH progression, Morton sorting, initialization, optimizer,
loss, learning-rate schedule, and final cleanup. The candidate adds a renderer
transform and a separate native FusedAdam for the global field. It must not put
the field in the native per-Gaussian optimizer, whose groups are pruned/extended.
Audit callbacks only observe state. No warmup, capacity cap, or early stopping.

Main DSEC suite: city_05_a 120-143, city_08_a 120-143, city_09_a 260-283;
seeds 3, 4, 5; 30,000 steps, batch size 1, 360x270. Same train/test split and
train-only 32,768-point stereo input as v3. Native initialization replaces the
previous hand-initialized temporal/scaling/opacity parameters. All native density
thresholds and camera-extent rules are retained, even for the narrow stereo rig.
This is a custom data adapter, not an official DSEC benchmark implementation.

External transfer: coffee_martini, the native PlenopticVideoBlender loader,
full 1352x1014 resolution, 5,100 training images, all 300 test images, original
589,789-point cloud, 30,000 steps, batch size 4, seed 0. Compare a new baseline
run to the field; the prior archived native run is context only. Its verified
lossless float32 disk cache may be used identically for both arms.

Each arm runs in its own process. Initial tensor hashes must match within pairs.
The native random sampler remains unchanged: density-dependent random-number
consumption can cause later image orders to differ, as in the original code.
The same seed is therefore not a claim of identical per-step samples.

Primary quality: native midpoint inference, official 8-bit quantization, mean
per-image PSNR, SSIM and VGG LPIPS. Also report float metrics and DSEC Gauss32
exposure inference. Candidate training uses one native render per sampled image.
Record time, final Gaussian counts, actual callback counts and source hashes.
Equal steps do not imply equal time or equal final capacity under native density
control. DSEC stability requires >=8/9 positive PSNR pairs, positive mean in each
clip, and no overall SSIM/LPIPS regression. A coffee result is one seed/scene and
cannot establish broad stability on its own. Report all outcomes, including losses.

Infrastructure smoke checks may use fewer steps and skip evaluation. Verify
zero-field image/gradient equality and exercise density/opacity/Morton callbacks.
If resource or infrastructure failure prevents completion, record it explicitly.
