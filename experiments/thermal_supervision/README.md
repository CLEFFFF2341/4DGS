# CTG-GS native thermal refinement

Current implementation: a runnable static, fixed-topology post-density refinement adapter using the real FasterGS4D CUDA rasterizer. It freezes native RGB parameters, uses independent RGB/thermal cameras and linear exposure integration, and commits guarded thermal geometry corrections. It is not the full native training loop or a validated dynamic-scene method.

V1 (`engine.py`) matches geometry and appearance source improvements. Controlled tests found that this rejected every correct geometry case. V2 (`profiled.py`) first fits the thermal head on the source for a fixed budget, holds it fixed during geometry search, then requires independent thermal-view improvement plus local/global/anchor RGB protection. Only the exact witnessed head and geometry offset are committed. No persistent thermal optimizer is used; do not attach an external optimizer to the thermal head or mutate the model concurrently. Rejection preserves model tensors and gradients; diagnostic counters and newly established anchors may change.

- [Implementation results](../../research/thermal_supervision_20261029/IMPLEMENTATION_RESULTS_zh.md)
- [Independent implementation review](../../research/thermal_supervision_20261029/IMPLEMENTATION_REVIEW.md)
- [Original proposal](../../research/thermal_supervision_20261029/PROPOSAL_zh.md)
- [Literature and novelty limits](../../research/thermal_supervision_20261029/LITERATURE.md)

## Run tests and controlled probes

From repository root, in PowerShell:

```powershell
$env:PATH = 'E:\4D_Gaussian\.conda\nerficg\Library\bin;' + $env:PATH
& ./.conda/nerficg/python.exe -m unittest discover -s experiments/thermal_supervision -t . -p 'test_*.py' -v
& ./.conda/nerficg/python.exe -m experiments.thermal_supervision.probe --engine v1 --out output/thermal_supervision/new_v1 --steps 3 --seeds 0 1 2
& ./.conda/nerficg/python.exe -m experiments.thermal_supervision.probe --engine v2 --out output/thermal_supervision/new_v2 --steps 3 --seeds 10 11 12 13 14
& ./.conda/nerficg/python.exe -m experiments.thermal_supervision.probe --out output/thermal_supervision/new_stress --steps 3 --seeds 10 11 12 --cases cold_start thermal_only_shift
```

Output directories must be new. CUDA tests skip without CUDA. Probe scenes contain 20 known Gaussians; targets and predictions use the same renderer. RGB guards integrate three camera samples at gain 0.25; evaluation uses an unseen sharp camera. This has no realistic sensor noise, thermal lag or rolling shutter and is not evidence of real-data gains. Test-camera observations only report metrics, never select candidates.

## Calibrated-data entry point

```powershell
& ./.conda/nerficg/python.exe -m experiments.thermal_supervision.run --checkpoint path/to/prefix.pt --manifest path/to/manifest.json --out output/thermal_supervision/refine --audit-only
```

Remove `--audit-only` to execute the manifest's finite transactions. Audit validates inputs without rendering or saving. Execution writes `checkpoint.pt` and `transactions.json`. Resume using that checkpoint, identical manifest/arrays and a new output directory. Engine config, anchors, next transaction and CPU/CUDA RNG are restored; data hashes reject changed observations. Native RGB optimizer is not resumed because it remains frozen. Use a native raw prefix, not a later N1b model with learned acceleration residuals.

Manifest schema illustration (add actual witness/guard rows referenced below):

```json
{
  "schema": 1, "units": "linear_sensor", "static_scene": true,
  "observations": [{
    "id": "source_ir", "viewpoint_id": "camera_0", "sensor": "thermal",
    "split": "train", "calibrated": true, "timestamp_seconds": 0.5,
    "target": "source_ir.npy", "weight": "source_weight.npy",
    "noise": 0.04, "gain": 1.0, "bias": 0.0,
    "exposure": [{"model_time": 0.5, "weight": 1.0,
      "camera": {"w2c": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "width": 64, "height": 48, "fx": 65, "fy": 65,
        "cx": 32, "cy": 24, "near": 0.1, "far": 100.0}}]
  }],
  "transactions": [{"source": "source_ir", "witnesses": ["witness_ir"], "guards": ["guard_rgb"]}]
}
```

Paths are inside the manifest directory. Targets are floating `.npy` CHW (1 thermal or 3 RGB channels), weights 1HW; noise is positive in calibrated sensor units. World-to-camera poses use positive forward Z and the checkpoint coordinate frame/scale. Intrinsics match each sensor's native grid. Exposure poses/times are externally measured. Positive thermal radiance uses softplus SH0: do not feed colorized IR, raw Celsius or gamma RGB as linear radiance. Declaring `calibrated` does not perform calibration. Masks/noise must be fixed independently of candidate fitting.

Only selected training arrays are opened. Distinct IDs alone are insufficient: physical thermal baseline and RGB support on changed projections are checked. Aggregate support does not prove every primitive observable. Darkness, occlusion, calibration error and complex thermal fields remain limitations. No densification or dynamic residual adapter is included. See the result report before interpreting synthetic PSNR.
