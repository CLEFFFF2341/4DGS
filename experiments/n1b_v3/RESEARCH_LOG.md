# Research ledger — 2026-10-26

The active objective is real held-out gains over matched ordinary Gauss. Original
v1/v2, production sources and previous checkpoints are untouched. All candidate
runs are in `output/n1b_v3/`; the training script refuses existing output folders.

## A. Real-gradient and support integration checks

`results/gradient_audit_01.json`: 16 fixed training observations across initial
and refined day/night development models. Compare actual MSE-loss gradients with
native Gauss64 branch AD. Gauss8 group-relative L2 error median 0.001676, max
0.04392. Gauss1 max is 0.37575. Gauss32/64 group max is 0.007858, so the dense
reference is a convergence diagnostic, not a mathematically exact derivative.
No moving-culling-boundary term is recovered.

`results/schedule_audit_01.json`: budget-8 partitions retain only screen-contribution
weighted temporal edges, so they avoid v2's full overflow fallback. Weighted
quantile edges improve image error in 14/16 and gradient error in 12/16 cases;
median error ratios 0.6404 and 0.5024, median measured cost ratio 1.493. Top-edge
selection is also recorded and has no consistent advantage.

Real training `dev02_c120_s0` and `dev02_n120_s0` disproves a useful quality gain:
weighted-edge PSNR relative to Gauss8 is about -0.00056 / -0.00309 dB. Gauss16
also does not materially improve the same training. **No sampling-only success.**

## B. Temporal representation development

These experiments change the time representation and keep ordinary Gauss for
rendering; their benefits must not be called quadrature improvements.

- `dev01_*`: fixed-pivot center parameterization is inconsistent. Monotone time
  warp, global quadratic translation and affine curvature offer small gains only.
- `dev02_*`: eight shared affine time anchors at LR 1e-4 gain ~0.20 dB day and
  ~0.074 dB night using v2's scale-aware/MSE optimizer. LR 1e-3 is a clear failure
  in both clips (preserved), so larger updates are not inherently beneficial.
- `results/radiometry_audit_01.json`: optimal training-only per-image affine RGB
  correction explains mean 2.71% / 0.52% of day/night training residual. No new
  photometric correction was added to the candidate or test images.

`core01_*` gives every arm the original native FusedAdam and L1+DSSIM loss, with
6,400 updates/renders. Candidate eight-knot affine center deformation adds 96
parameters and uses a 3e-5 LR. Day/native 29.6696 dB, candidate 30.2531; night
40.0371 vs 40.3816. A two-anchor linear field barely helps day and regresses
night; a single quadratic field is weaker. These are development ablations.

`rep01_*` repeats fixed 3e-5 candidate in existing c120/c260/n120 seeds 1/2.
All six improve: approximately +0.8821,+0.8256,+0.3082,+0.2952,+0.2811,+0.3375 dB.
These are replications on already used clips, not untouched final datasets.

`time01_*`: same 12-second allowance, separate restarts, candidate improves
development day/night by +0.2660/+0.2196 dB. Actual steps, times and final-step
overrun are recorded. No universal speed claim.

`results/development_audit_01.json`: independent float LPIPS and dense-integral
check on core day/night and c260 seed1. LPIPS improves 0.24036→0.22364,
0.24108→0.23516, and 0.29206→0.27579 respectively. Checkpoint hashes match.

## C. Frozen confirmation

See `CONFIRMATION01_PROTOCOL.md`, `HELDOUT_PLAN.md`, and
`confirmation01_frozen.json`. Three new windows, two new sequences, seeds 3/4/5;
both equal-render and equal-time comparisons. The candidate was frozen before
these results, and source hashes are enforced on every case. All cases are to
run even if an early case fails. Final success is not declared until the complete
metric, LPIPS, timing and hash audit satisfies the declared gate.

Completed: all 18 budget/seed pairs and independent audits are saved. The
predeclared gate passed. Equal-render PSNR/SSIM/LPIPS improve in 9/9 pairs;
PSNR mean +0.29588 dB. Equal-time PSNR improves in 8/9, mean +0.18689 dB;
equal-time LPIPS worsens by 0.002056 on average and is not hidden.

Predeclared Gauss8 controls and 30,000-step checks also retain PSNR gains. See
`RESULTS.md` for the exact budgets, limits and negative cases. Post-confirmation
minimum-gain numerical repeats are separate records, never replacements of the
original confirmation or additional untouched test evidence.

The minimum-gain night/seed4 pair was repeated twice with reversed arm order.
Original +0.09520 dB; repeats +0.02824/+0.01129 dB. Signs remain positive but
the magnitude is numerically sensitive. This limitation is included in the final
report; no result was replaced and no further candidate tuning was performed.
