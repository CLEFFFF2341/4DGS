# N1b continued research, 2026-10-26

Objective: reproducible held-out real-image gains over a strong matched Gauss
control. User authorizes repeated small and real-data experiments. No fixed
iteration count is assumed, and unsuccessful runs remain part of the record.

## Evidence and scope

N1b-v1's stochastic proxy backward had heavy tails. N1b-v2's temporal-support
partition works on narrow synthetic pulses, but falls back to global Gauss on
all measured DSEC training steps. Existing real-image quadrature convergence
and matched midpoint controls suggest an integration accuracy bottleneck is
unlikely. This is not a proof of gradient convergence.

Stage A audits actual training-loss branch gradients and a budgeted deterministic
interval proposal. Stage B may change the parameterization of native temporal
structure if Stage A does not support an integration-only route. Such a change
must be called a successor/reparameterization, not evidence that v1 succeeded.

## Data separation

- Development: existing DSEC city_04_c frames 120--143 seed 0, and city_09_a
  frames 120--143 seed 0. Their old held-out metrics have already been inspected;
  they are development validation, never untouched final tests.
- Existing remaining clips/seeds can be used for replication and diagnostics,
  but are not claimed as independent unseen datasets.
- Before confirmatory runs, save the exact candidate, code hashes, budgets,
  fresh seeds, additional real clips, metrics and success rule in a separate
  frozen protocol. Once used to choose a variant, a split becomes development.
  Further changes require a new untouched confirmation set.
- Train/test initialization separation in the existing data adapters is retained.
  No test pixels, test depth, test loss or test-driven stopping during training.

## Controls and success

All compared arms share initialization, observations, capacity, train order,
optimizer/loss unless that difference is explicitly the tested intervention.
Any general optimizer/loss change must also be given to the Gauss control.
Compare Gauss1/2/4/8 as appropriate, and retain original FasterGS core as a
stronger contextual baseline. Original core is fixed capacity, not the full
Trainer with densification. Equal steps/renders and actual wall time are distinct.

Report per-clip and per-seed PSNR, SSIM and LPIPS, not only pooled image means.
The working minimum practical quality target is +0.2 dB mean PSNR across fresh
clip/seed pairs, positive mean gain on each confirmation clip, no substantial
SSIM/LPIPS regression, and repeatability beyond run noise. This target is a
research gate, not a guarantee. Claim scope follows the evidence, including
capacity and runtime. Speed claims require controlled repeated timing.

## Integrity

New output directories refuse overwrite. Record raw metrics, training renders,
updates, time, parameter count, initial/model/code hashes, failures and candidate
definitions. All current source/checkpoints and prior negative results remain.
No production source, security settings, or blocked binary policies are changed.
