# Joint fitting repair: frozen confirmation protocol

2026-10-25. This file is written after seed0 exploration and before running seeds
6–11. Archived seeds0–5 and the new seed0 ablations informed the repair. They are
not independent confirmation. All exploratory failures remain under
`results/joint_audit/exploration_*.json`.

## Diagnosis and intervention

The archived joint probe updates world-space XYZ and log temporal scales with
the same Adam LR=.01. Its 8x8 central ROI discards the edges of the 32x32 image
already rendered by CUDA. Repeated, identically colored, vertically elongated
Gaussians observed through one integrated exposure provide weak geometry cues.
At seed0 initialization, finite-difference local sensitivity gives a condition
number about 2.43e5 and temporal-width columns explain about 99.26% of depth-column
energy. These are local numerical measurements, not a proof of global ambiguity.

The repair uses all pixels, position = initial_position + L * normalized_offset,
where L is the Cholesky factor of the initial conditional spatial covariance,
and independent Adam rates .01 (normalized position), .05 (log temporal width).
L is detached/frozen; all XYZ coordinates and temporal widths remain trainable.
There is no ground-truth anchor, geometry penalty, frozen depth, or test-view
selection of updates. The transform does retain an initialization-dependent
optimization preference, so incorrect-depth initialization must be tested.

The integrator's detached branch-AD semantics remain unchanged. Numerical
finite differences show omitted moving-culling-boundary terms; this repair does
not claim to supply the full derivative of the discontinuous renderer.

## Confirmation

- Seeds6–11, original fixture generator (87000+seed), one training camera,
  full 32x32 images, unchanged known shutter [0.3,0.7].
- Per arm: 3-second local optimization cap, including parameter/optimizer
  construction, quadrature, CUDA forward/backward, updates, synchronization and
  in-loop diagnostic collection. Last over-budget update is restored; its wall
  time is reported. Targets and independent evaluation are outside the cap.
- Partition32: legacy raw coordinates/rates, normalized coordinates with equal
  .01 rates (ablation), normalized coordinates with .01/.05 rates (repair).
- Gauss64 and Gauss128 receive the identical repair, pixel set, targets and cap.
- Held-out cameras at x=.025, -.15, +.15, none used to train/stop/select steps.
- Report both full-image and original central-ROI check MSE, position XYZ MSE,
  temporal log-scale MSE, accepted steps and actual times. Full-frame MSE must
  not be compared directly to historical 8x8 MSE without using the ROI metric.
- Compare repair against legacy, and against the best measured repaired Gauss
  arm per case. Report every failure; image improvement alone is not proof of
  geometry recovery. Geometry must also improve over initialization to claim it.

## Additional checks, fixed before confirmation

- Same-ROI ablation on seeds6/7: original observation set, legacy vs repaired
  optimizer, so increased image coverage is not conflated with optimizer changes.
- Repeat repaired Partition32 and Gauss64/128 on seeds6/7 twice more to expose
  timer variability. These repeats are not new independent scenes.
- Stress seeds6–9 with initial XYZ offsets (.015,.02,.06), one full-frame train
  view and the same repaired Partition32/Gauss64/Gauss128 arms.
- Equal-step check: seed6, 160 accepted updates, repaired Partition32 and
  Gauss128, report measured time rather than claiming equal-time comparison.
- Independent event-split order8/16 evaluation convergence at truth and final
  fitted models for seeds6/7. Reference is valid for these stationary fixtures.
- Unit checks: spatial coordinate unit invariance, conditional covariance,
  all-coordinate trainability, terminal rollback, and pixel-selection regression;
  existing partition tests must continue passing.

No parameter search is performed on confirmation seeds. The result may be a
useful image-fitting repair without uniquely recovering geometry. Real-data
reconstruction, unknown exposure, trainable topology/colors/rotations, and
million-Gaussian scaling are outside this experiment; the native Trainer is
still a single-timestamp renderer and is not silently replaced by this probe.
