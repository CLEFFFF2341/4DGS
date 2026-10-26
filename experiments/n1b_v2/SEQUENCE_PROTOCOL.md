# Additional temporal observations: protocol

2026-10-25, written before running this experiment. The single-long-exposure
confirmation is complete and exposed an important limitation: six repaired
fits improve integrated test images, but only one improves position MSE over
initialization, and two worsen instantaneous test images relative to legacy.

This follow-up changes the observations to address that limitation explicitly.
It is not evidence that a new optimizer can create missing information, nor an
integrator win at equal data. The original long-exposure failures stay recorded.

- XYZ initialization offsets (.015,.02,.06), temporal widths 1.7 times truth.
- All XYZ and widths train; other parameters, including time centers, are fixed,
  as in the original experiment.
- A fixed temporal frame grid of .002 with known .002-duration shutters. Select
  the nearest frame and its two neighbors around INITIAL model time centers.
  This uses neither true widths nor test errors. Initial time centers are known
  in this restricted experiment; no claim is made about estimating unknown ones.
- Batch one temporal offset across all selected center groups per update, cycling
  offsets -1/0/+1, so the staggered case updates all primitives from informative
  frames. All rendering/gradient/optimizer work is charged to the clock.
- Compare the repaired optimizer with one long exposure + Partition32, versus
  the additional frames + Partition8, versus the same additional frames +
  Gauss8. Local optimization budget is 6 seconds for each; final overrun rolls
  back. Target generation and independent evaluation are outside the cap.
- Evaluate position XYZ/log-width error, instantaneous true-peak images at three
  unseen cameras, and exposures halfway between training centers at an unseen
  camera. Neither true parameters nor evaluation images control training.
- Seeds12/13 are first checks. Freeze the same design for seeds14–17 regardless
  of outcome; report all failures and do not tune on those seeds.
- The native stationary event reference integrates all relevant temporal and
  pixel-alpha boundaries. Identical float32 timestamps may be merged because
  the CUDA timestamp argument is float; do not merge distinct timestamps.

This is still a synthetic small-model study. Real observations require measured
shutter metadata; fps alone must not be used to invent an exposure duration.

Post-confirmation diagnostic (no retuning): seeds14/15 were also compared at
160 updates for every arm to check the update-count confound. This is not an
equal-time comparison; measured time is reported. Order8/16 checks were added
for truth and fitted models at every held-out shutter of seeds14/15 (27 checks).
