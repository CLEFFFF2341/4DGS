# Confirmation 01: shared temporal center deformation

Frozen 2026-10-26, before any candidate metrics on the reserved clips.

## Candidate and attribution

The candidate adds an eight-knot, linearly interpolated, shared affine correction
to native Gaussian centers: x_input(t) = x_native_mean + A(t)x_native_mean+b(t).
Each knot stores a 3x4 matrix, initially zero: **96 extra parameters** relative
to 2,129,920 native parameters (32,768 Gaussians). The native conditional velocity,
opacity, covariance, SH and complete occlusion renderer remain in use. This is
a center-deformation model, not a claim of a rigid physical camera trajectory
or covariance pushforward. It is a successor temporal representation, **not an
improved N1b quadrature formula**. Both candidate and baseline use ordinary Gauss.

Candidate: `affine_spline:1:0.00003`. Control: `native:1:0`.
Original FasterGS FusedAdam, original L1/DSSIM loss and native parameter learning
rates are identical. Extra field uses FusedAdam LR 3e-5. The original spatial
LR schedule continues from shared warmup step 1200. No other regularizer, new
observations, learned image corrections, density changes or test-driven stopping.

## Data

Exactly the three clips in HELDOUT_PLAN.md, seeds 3,4,5. Two new capture sequences
plus one new window of an already used night capture. 360x270, 24 times per clip,
both calibrated cameras, hold out local indices 2,6,10,14,18,22. Train-only stereo
initialization, 32,768 Gaussians, common 1200-step Gauss8 warmup from v2. All arms
load identical per-pair checkpoints and use seed+123 observation order.

## Budgets, ordering and evaluation

Two independently restarted experiments per clip/seed:

1. Equal renders/updates: 6,400 steps, one native render per update.
2. Equal wall-clock allowance: 12 seconds, at most 30,000 steps. CUDA synchronize
   after every update and stop before beginning another once time is reached.
   One final-step overrun is recorded; no extrapolated timing or free field cost.

Control first for even seeds, candidate first for odd seeds. Initialization,
model construction, I/O and evaluation are excluded identically from incremental
training time. No other CUDA training runs concurrently; desktop graphics remain
active, so timing is a local repeated-budget result, not a universal speed claim.

All models evaluated with the same Gauss32 integration over the real exposure.
Report per-view and per-clip/seed float PSNR, SSIM, fixed ROI; independent LPIPS
and Gauss32/64 convergence audit follows. No best-checkpoint selection.

## Fixed decision rule

Equal-render candidate must achieve mean gain >=0.2 dB across the nine pairs,
positive mean gain for each clip and positive gains for at least 8/9 pairs.
Mean SSIM must not decrease and LPIPS must not increase on each clip.
Equal-time mean PSNR gain must be positive overall and for each clip; report
all seed failures and actual update counts. This is a practical research gate,
not a multiple-testing-adjusted universal significance claim.

Run all nine planned pairs even if early results fail. Do not change the
candidate or reject a scene after seeing outcomes. A failure makes this a
negative confirmation, and later revisions need a new reserved set.

## Limitations

Fixed-capacity small image reconstruction; no full default Trainer, densification,
unknown camera calibration or instantaneous clear-RGB truth. Additional field
parameters and runtime are reported. A win validates this temporal structure
only within the tested setup; it does not revive the old stochastic proxy.
