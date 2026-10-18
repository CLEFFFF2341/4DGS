# FasterGS4D coffee_martini deployment

Paper reproduction audit: the coffee_martini workflow is complete, but paper-level numerical reproduction is **not verified**. Table 11 covers three real scenes and does not report real-scene SSIM/LPIPS. See `PAPER_REPRODUCTION_AUDIT.md` and `metric_audit.json`; completion must not be interpreted as matching every paper metric.

Completed on 2026-10-18: all 30,000 training iterations, all 300 held-out cam00 frames, saved-checkpoint reloading and finite-parameter validation, full PSNR/SSIM/LPIPS-VGG evaluation, and a verified 300-frame 30 fps dynamic MP4. Measured metrics: PSNR 26.321117 dB, SSIM 0.893975, LPIPS 0.212569. See `RESULTS.md` and `results.json` for final outputs and measurements.

## Official sources and pinned code

- NeRFICG: https://github.com/nerficg-project/nerficg, commit `e458ac6e41810ad6d9405687c1e698fadd28dac5`.
- FasterGS4D: https://github.com/nerficg-project/faster-gaussian-splatting/tree/FasterGS4D, commit `3dbbb985c5e0f29825d0960563f3c39f29eb3987`.
- Dataset: https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0.
- Preprocessing: https://github.com/fudan-zvg/4d-gaussian-splatting/blob/main/scripts/n3v2blender.py, original saved in `upstream/`; source commit recorded in `upstream/4dgs-commit.json`.
- Experimental settings: https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf, Appendix C.5.

## Required experiment settings

| Setting | Value |
|---|---|
| Training and test image resolution | 1352 x 1014 |
| Loader image scale | 0.5 of native 2704 x 2028 PNGs |
| Time window | 0 <= time < 10 seconds |
| Frame timestamps | Original frame index / 30; loader normalizes selected range to [0, 1] |
| Frames per camera | 300, indices 0000..0299; all 18 videos verified using ffprobe |
| Total split sizes | 5100 training images (17 cameras), 300 test images (cam00) |
| Test split | cam00, all selected frames |
| Train split | All other valid camera streams; never interpolate missing camera IDs |
| Iterations | 30,000 |
| Batch size | 4 |
| SH degree | 3 |
| Initialization | Official preprocessing dense point cloud, not random replacement |
| Random seed | 0 |
| Evaluation | Official NeRFICG saved 8-bit images, PSNR, SSIM, LPIPS-VGG |
| Dynamic video | Held-out cam00 sequence, original temporal order, 300 frames at 30 fps |

The model configuration is `configs/fastergs4d_coffee_martini.yaml`, derived from the official branch's example. GUI is disabled and data preloading is disabled to fit this machine's approximately 48 GB RAM. These operational choices do not change resolution, frame count, split, batch size, or optimization schedule. No claims about completed training or measured metrics should be made without checkpoint and metric files.

## Environment and manual prerequisite installers

Independent Conda prefix: `E:\4D_Gaussian\.conda\nerficg` (Python 3.11).

The earlier named environment `C:\Users\CLEF\miniconda3\envs\nerficg` became corrupt after interruption. It is preserved and is not used by this deployment.

- CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive
- MSVC 2022 Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe

Select Desktop development with C++, MSVC v143, Windows SDK, and C++ CMake tools. Install MSVC before CUDA. Existing NVIDIA driver need not be replaced.

Official Python dependencies were extracted unchanged to `requirements-official.txt`. This terminal receives proxy HTTP 403 for PyPI and download.pytorch.org. The working headless runtime was instead installed from Anaconda's official main channel: PyTorch 2.8.0 GPU CUDA128 build, torchvision 0.23.0 CUDA128, setuptools 80.10.2, pybind11-global 2.13.6, and the required scientific/runtime modules. `plyfile` 1.1.5 was built from its upstream repository, commit `9191fccd3a83fcac1f1154b8cb8c756f36897144`, with no code changes. Optional GUI and other-method packages are not required for this headless experiment.

The installed CUDA Toolkit is 12.8.61 and MSVC is 14.44.35207. The Anaconda PyTorch binary reports CUDA 12.8 and cuDNN 92600 (upstream's tested environment reports cuDNN 91002); its Conda-managed CUDA runtime dependencies include 12.9 libraries. This packaging difference is recorded explicitly. `gpu-preflight.log` records successful RTX 4090 tensor computation. Exact environment packages are recorded in `current-conda-explicit.txt` and `current-pip-freeze.txt`.

Anaconda's development headers and libraries reside under `Library/include` and `Library/lib`; `run_training.ps1` adds those paths to MSVC INCLUDE/LIB. A package test filename exceeded Windows MAX_PATH during extraction; the SHA256-verified archive was extracted with Windows extended-length paths, without changing system policy or package contents.

If the original wheel indexes become available, the unmodified official environment requirements can also be installed with:

```powershell
.\.conda\nerficg\python.exe -m pip install -r deployment/requirements-official.txt
```

## Reproducible execution

```powershell
.\.conda\nerficg\python.exe deployment/download_dataset.py
# Extract downloaded FFmpeg and COLMAP ZIPs into tools/ first.
.\.conda\nerficg\python.exe deployment/prepare_coffee.py
.\.conda\nerficg\python.exe deployment/cache_images.py
powershell -ExecutionPolicy Bypass -File deployment/run_training.ps1
```

`prepare_coffee.py` retains the official camera transform, split, native extraction resolution and COLMAP dense initialization. Its wrapper adapts Windows path handling, replaces privileged symlinks with byte-identical copies, uses NumPy-compatible binary buffer APIs, reuses verified complete frame sequences on restart, and retains the COLMAP workspace for audit. COLMAP 4.2.1 is used with `--ImageReader.single_camera 1` to match the upstream known-pose model's single shared PINHOLE camera. `dataset_archive.json` records the verified archive size and checksums. `video_metadata.json` and `dataset_settings.json` are produced only after the relevant steps execute.

The completed initial point cloud has 589,789 vertices. Full dataset validation is recorded in `dataset-verify.log` and `dataset_settings.json`. Following interruption, existing dense maps were retained and `patch_match_stereo` was resumed directly; this avoids COLMAP's failure when re-running image_undistorter over existing outputs. Final fusion is recorded in `fusion.log`.

LPIPS uses the original torchvision VGG16 ImageNet weights republished by the upstream timm project at https://huggingface.co/timm/vgg16.tv_in1k. Their pinned file hash and unchanged feature-tensor verification are recorded in `lpips_weights.json`. Classifier keys are converted back to torchvision layout and strictly loaded before caching. The GPU identity-image LPIPS score is exactly 0.0. This preserves the official VGG LPIPS metric; no random or substitute feature weights are used.

Training, evaluation and video commands have not run successfully until their corresponding logs and artifacts exist. Inspect `train.log`, `evaluate.log`, `video.log`, the final checkpoint and `metrics_8bit.txt` before reporting results.

## Lossless data loading cache

An initial diagnostic run measured approximately 2.6 iterations/second while repeatedly decoding and resizing PNGs. It was stopped at about 220 iterations and is not a completed experiment; its log is `train-uncached-diagnostic.log`. The complete experiment restarts from seed 0 for all 30,000 iterations.

`cache_images.py` uses the unchanged official `ImageData._load()` implementation with image scale 0.5 and area resampling, then stores its float32 outputs without quantization. All 5400 stored images were compared with their original loader tensors and passed exact equality checks. The cache occupies 88,836,825,600 bytes; source image hashes and the complete validation manifest are in `image_cache.json`.

`run_with_cache.py` runs the unchanged official train/inference entry points, replacing only matching coffee_martini RGB loads with these lossless tensors through memory mapping. It rechecks eight images against the original loader under the initialized training runtime. Resolution, timestamps, train/test split, sampler, losses, model, optimization settings, seed and iteration count are unchanged. Cache generation time is recorded separately from training time.
