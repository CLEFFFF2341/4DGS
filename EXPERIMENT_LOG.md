# Ex4DGS reproduction log

- Started: 2026-09-22 (Asia/Shanghai)
- Repository: `E:\4DGS`
- Constraint: keep algorithm source code unchanged; use official datasets and official pretrained models.

## Command log

### 1. Read PDF workflow instructions

```powershell
Get-Content -Raw -LiteralPath 'C:\Users\CLEFFFF\.codex\plugins\cache\openai-primary-runtime\pdf\26.904.11930\skills\pdf\SKILL.md'
```

Result: succeeded. The PDF workflow requires complete text inspection plus visual review of relevant pages.

### 2. Load bundled PDF dependencies

Called the Codex workspace dependency loader.

Result: succeeded. Bundled Python and Poppler-compatible binary paths are available.

### 3. Inventory local papers and PDF tools

```powershell
Get-ChildItem -LiteralPath 'E:\4DGS\Papers' -File -Recurse
Get-Command pdfinfo,pdftoppm
```

Result: succeeded. The target paper is `E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf` (42,351,774 bytes). Both `pdfinfo` and `pdftoppm` are available from the bundled Poppler runtime.

### 4. Verify experiment log creation

```powershell
Test-Path -LiteralPath 'E:\4DGS\EXPERIMENT_LOG.md'
Get-Content -Raw -LiteralPath 'E:\4DGS\EXPERIMENT_LOG.md'
```

Result: succeeded; the log exists.

### 5. First attempt to append PDF extraction notes

Action: attempted to append new sections with `apply_patch` using a Chinese section header that was not present in the existing English log.

Result: failed validation because the expected context `## 4. 创建实验日志` did not match the actual heading `### 4. Verify experiment log creation`. No file content was changed.

Fix: read the current log content, then append using the exact existing context.

```powershell
Get-Content -LiteralPath 'E:\4DGS\EXPERIMENT_LOG.md' -Raw
```

Result: succeeded.

### 6. Inspect PDF metadata and try Poppler text extraction

```powershell
$poppler='C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
$pdf='E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'
$tmp='C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\pdfs'
New-Item -ItemType Directory -Path $tmp -Force
& "$poppler\pdfinfo.exe" $pdf
& "$poppler\pdftotext.exe" -layout $pdf "$tmp\ex4dgs.txt"
rg -n -i "dataset|N3V|Technicolor|camera|frame|time|pretrain|render" "$tmp\ex4dgs.txt"
```

Result: `pdfinfo.exe` identified a 19-page PDF. The bundled Poppler directory does not contain `pdftotext.exe`, so no text file was generated and the subsequent `rg` command also failed.

Errors: PowerShell could not find `pdftotext.exe`; `rg` reported that `ex4dgs.txt` did not exist.

Fix: use the bundled Python runtime and `pypdf` for text extraction; retain `pdftoppm.exe` for rendered-page visual inspection.

### 7. First pypdf extraction attempt

```powershell
Get-ChildItem -LiteralPath 'C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin' -File
Get-Item -LiteralPath 'E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'
& 'C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "from pypdf import PdfReader; p=PdfReader(r'E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'); print('pages',len(p.pages)); [print(f'--- PAGE {i+1} ---\n'+(page.extract_text() or '')[:800]) for i,page in enumerate(p.pages)]"
```

Result: confirmed that the PDF has 19 pages and extracted the first three page previews.

Error: printing page 4 raised `UnicodeEncodeError` because the active Windows GBK console encoding could not represent a mathematical bold character.

Fix: set `PYTHONIOENCODING=utf-8` for the Python process and rerun.

### 8. Retry pypdf extraction with UTF-8 output

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "from pypdf import PdfReader; p=PdfReader(r'E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'); print('pages',len(p.pages)); [print(f'--- PAGE {i+1} ---\n'+(page.extract_text() or '')[:800]) for i,page in enumerate(p.pages)]"
```

Result: succeeded for all 19 pages. Pages 7–14 contain the principal experiments, dataset notes, supplementary experiments, and detailed results. Page 12 states that the Technicolor Birthday experiment uses frames 50–169 (120 frames) and the point cloud from frame 50 for initialization.

### 9. Extract the complete experiment sections from pages 7–14

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "from pypdf import PdfReader; p=PdfReader(r'E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'); [print(f'===== PDF PAGE {i+1} =====\n'+(p.pages[i].extract_text() or '')) for i in range(6,14)]"
```

Result: succeeded. Reproduction-relevant facts extracted from the paper:

- N3V: six indoor scenes, 18–21 cameras, 2704×2028, 300 frames; train/evaluate at half resolution; hold out the center camera.
- Technicolor: synchronized 4×4 camera array, 2048×1088; hold out the camera at row 2, column 2; evaluate Birthday, Fabien, Painter, Theater, and Train at full resolution.
- Initialization uses only the COLMAP point cloud from the first frame. Time interval and initial duration are 10; duration increases by 10 every 400 iterations.
- The appendix additionally specifies Birthday frames 50–169 (120 frames, point cloud from frame 50), Train frames 170–269 (100 frames, point cloud from frame 170), and a 1,000-frame Flame Salmon experiment.

### 10. Render and visually inspect the relevant PDF pages

```powershell
$poppler='C:\Users\CLEFFFF\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
$pdf='E:\4DGS\Papers\Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf'
$out='C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\pdfs\ex4dgs-page'
& "$poppler\pdftoppm.exe" -f 7 -l 8 -png -r 144 $pdf $out
& "$poppler\pdftoppm.exe" -f 12 -l 12 -png -r 144 $pdf $out
Get-ChildItem -LiteralPath 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\pdfs' -Filter 'ex4dgs-page-*.png' | Select-Object Name,Length,LastWriteTime
```

Result: succeeded. Generated `ex4dgs-page-07.png` (409,087 bytes), `ex4dgs-page-08.png` (855,143 bytes), and `ex4dgs-page-12.png` (984,540 bytes). Visual inspection confirmed the extracted camera/resolution/frame statements and the figures/tables are legible and consistent with the extracted text.

### 11. Locate official dataset and pretrained-model sources

Action: searched the official Ex4DGS repository/release, the official Meta Neural 3D Video repository/release, and the official InterDigital Light-Field dataset page.

Result: the Ex4DGS README points to release `v0.1` for pretrained models. N3V provides public scene archives through its GitHub `v1.0` release. Technicolor requires requesting access by email from InterDigital, so it cannot be downloaded automatically without user-provided access.

Official pages:

- `https://github.com/juno181/Ex4DGS/releases/tag/v0.1`
- `https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0`
- `https://www.interdigital.com/data_sets/light-field-dataset`

### 12. Query official GitHub release APIs for asset names and sizes

```powershell
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/juno181/Ex4DGS/releases/tags/v0.1' -Headers @{ 'User-Agent'='Codex-Reproduction' }
$release | Select-Object tag_name,name,published_at,html_url
$release.assets | Select-Object name,size,download_count,browser_download_url | Format-Table -AutoSize
$n3v = Invoke-RestMethod -Uri 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0' -Headers @{ 'User-Agent'='Codex-Reproduction' }
$n3v | Select-Object tag_name,name,published_at,html_url
$n3v.assets | Select-Object name,size,download_count,browser_download_url | Format-Table -AutoSize
```

Result: succeeded. Ex4DGS provides 11 per-scene model archives. N3V provides six public scene archives. `cut_roasted_beef.zip` is the smallest single-file N3V scene archive (1,143,395,383 bytes); its official Ex4DGS model archive is 109,485,709 bytes. `flame_salmon_1` is split into four archive parts, so it is not selected for the first reproduction run.

### 13. Inspect repository instructions, configs, loaders, renderer, and preprocessing scripts

```powershell
$files = @(
  'README.md',
  'configs\N3V\n3v_base.json',
  'configs\techni\Birthday.json',
  'scripts\preprocess_all_n3v.sh',
  'scripts\preprocess_all_techni.sh',
  'scene\dataset_readers.py',
  'scene\__init__.py',
  'arguments\__init__.py',
  'render.py'
)
foreach ($f in $files) { "===== $f ====="; Get-Content -LiteralPath (Join-Path 'E:\4DGS' $f) -Raw }
"===== relevant references ====="
rg -n "duration|time_interval|frame|camera|video|poses_bounds|colmap|cam00|test|eval|resolution|source_path|model_path|configpath" 'E:\4DGS\configs' 'E:\4DGS\scripts' 'E:\4DGS\dataset_utils' 'E:\4DGS\scene' 'E:\4DGS\render.py' 'E:\4DGS\arguments'
```

Result: succeeded; console display was truncated after 32,759 tokens, so narrower follow-up inspections are required. Confirmed that `configs/N3V/n3v_base.json` uses `resolution=2`, `duration=-1`, `start_duration=10`, `time_interval=10`, `iterations=40000`, and loader `neural3dvideo`. The N3V preprocessing entry point extracts frames 0–299 and creates `colmap_0`; the loader reads `poses_bounds.npy` and per-camera frame folders.

### 14. Narrow inspection of dataset split, checkpoint loading, and rendering behavior

```powershell
rg -n -A 110 -B 15 "def readN3VInfo|def readTechnicolorInfo|sceneLoadTypeCallbacks" 'E:\4DGS\scene\dataset_readers.py'
rg -n -A 90 -B 20 "sceneLoadTypeCallbacks|load_iteration|train_cameras|test_cameras|model_path|source_path" 'E:\4DGS\scene\__init__.py'
rg -n -A 120 -B 20 "def render_sets|Scene\(|getTestCameras|iteration|skip_train|skip_test" 'E:\4DGS\render.py'
rg -n -A 120 -B 20 "class ModelParams|configpath|start_timestamp|end_timestamp|duration|eval" 'E:\4DGS\arguments\__init__.py'
Get-Content -LiteralPath 'E:\4DGS\scripts\pre_n3d_colmap.py' -Raw
rg -n -A 130 -B 20 "def getcolmapsinglen3d" 'E:\4DGS\dataset_utils\etc_utils.py'
```

Result: succeeded, but output was still partly truncated. A second line-range inspection was run:

```powershell
$p='E:\4DGS\scene\dataset_readers.py'
$lines=Get-Content -LiteralPath $p
for($i=430;$i -le 585;$i++){ '{0,4}: {1}' -f ($i+1),$lines[$i] }
$p='E:\4DGS\render.py'
$lines=Get-Content -LiteralPath $p
for($i=120;$i -lt $lines.Count;$i++){ '{0,4}: {1}' -f ($i+1),$lines[$i] }
$p='E:\4DGS\arguments\__init__.py'
$lines=Get-Content -LiteralPath $p
for($i=135;$i -lt $lines.Count;$i++){ '{0,4}: {1}' -f ($i+1),$lines[$i] }
```

Result: succeeded. The N3V loader always reserves `cam00` as the test view and uses all other cameras for training. It reads `colmap_<start_timestamp>/sparse/0`, all per-camera PNG frame folders, and `poses_bounds.npy`. The renderer reads the model's saved `cfg_args`, loads `point_cloud/iteration_<N>/point_cloud.ply`, and renders the held-out test cameras unless `--skip_test` is supplied.

### 15. First official model download attempts

```powershell
New-Item -ItemType Directory -Force -Path 'E:\4DGS\downloads','E:\4DGS\pretrained','E:\4DGS\data' | Out-Null
curl.exe -L --fail --retry 5 --retry-delay 3 -o 'E:\4DGS\downloads\cut_roasted_beef-model.zip' 'https://github.com/juno181/Ex4DGS/releases/download/v0.1/cut_roasted_beef.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip'
Get-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip' | Select-Object FullName,Length,LastWriteTime
```

Result: failed. `curl` reported `Recv failure: Connection was reset`; the file did not exist, so both verification commands also failed.

Fallback command:

```powershell
Get-Command gh,aria2c,wget.exe -ErrorAction SilentlyContinue | Select-Object Name,Source
if (Get-Command gh -ErrorAction SilentlyContinue) { gh auth status; gh release download v0.1 --repo juno181/Ex4DGS --pattern 'cut_roasted_beef.zip' --dir 'E:\4DGS\downloads' --clobber }
```

Result: none of these alternative download tools were installed.

Second fallback command:

```powershell
$ProgressPreference='SilentlyContinue'
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/juno181/Ex4DGS/releases/download/v0.1/cut_roasted_beef.zip' -OutFile 'E:\4DGS\downloads\cut_roasted_beef-model.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip'
Get-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip' | Select-Object FullName,Length,LastWriteTime
```

Result: the foreground command timed out while the spawned PowerShell process continued in the background. Initial inspection showed a 0-byte file; it later grew because the background process was still active.

Diagnostic commands:

```powershell
Get-ChildItem -LiteralPath 'E:\4DGS\downloads' -Force | Select-Object Name,Length,LastWriteTime
Get-Process -Name powershell,pwsh,curl -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime
curl.exe --http1.1 -I -L --max-redirs 5 'https://github.com/juno181/Ex4DGS/releases/download/v0.1/cut_roasted_beef.zip'
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/juno181/Ex4DGS/releases/tags/v0.1' -Headers @{ 'User-Agent'='Codex-Reproduction' }
$asset = $release.assets | Where-Object name -eq 'cut_roasted_beef.zip'
$asset | Select-Object id,name,size,url,browser_download_url,digest | Format-List
```

Result: direct `github.com:443` connection failed after approximately 21 seconds. GitHub API remained reachable and identified asset ID `202878157`, expected size `109485709`; no upstream digest was published.

### 16. Switch model download to the GitHub release-asset API

The first command tried to remove the failed file before retrying:

```powershell
Remove-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip' -Force -ErrorAction SilentlyContinue
curl.exe --http1.1 -L --fail --retry 5 --retry-all-errors --retry-delay 3 -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' -o 'E:\4DGS\downloads\cut_roasted_beef-model.zip' 'https://api.github.com/repos/juno181/Ex4DGS/releases/assets/202878157'
```

Result: command launch was rejected by the execution safety policy because it included file deletion. No command ran.

Fix: write to a new filename without deleting anything:

```powershell
curl.exe --http1.1 -L --fail --retry 5 --retry-all-errors --retry-delay 3 -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' -o 'E:\4DGS\downloads\cut_roasted_beef-model-v2.zip' 'https://api.github.com/repos/juno181/Ex4DGS/releases/assets/202878157'
Get-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model-v2.zip' | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model-v2.zip'
```

Result: the foreground wrapper returned after 30 seconds while the download process continued in the background. A subsequent resumable `curl -C -` started concurrently and corrupted the shared output file. Archive listing failed with `Damaged Zip archive`; hash/extraction failed because another process still held the file.

Process diagnosis:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'curl.exe'" | Select-Object ProcessId,CreationDate,CommandLine | Format-List
Get-CimInstance Win32_Process -Filter "Name = 'pwsh.exe'" | Where-Object { $_.CommandLine -match 'cut_roasted_beef-model' } | Select-Object ProcessId,CreationDate,CommandLine | Format-List
Get-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model-v2.zip' | Select-Object FullName,Length,LastWriteTime
```

Result: confirmed one lingering `curl.exe` and two wrapper `pwsh.exe` processes operating on the model downloads.

Fix and preservation of failed artifacts:

```powershell
Stop-Process -Id 28168,21808,29988 -Force
Start-Sleep -Seconds 1
Get-CimInstance Win32_Process -Filter "Name = 'curl.exe'" | Select-Object ProcessId,CommandLine
Move-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model-v2.zip' -Destination 'E:\4DGS\downloads\cut_roasted_beef-model-v2.corrupt.zip'
Move-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip' -Destination 'E:\4DGS\downloads\cut_roasted_beef-model.zero-byte.zip'
Get-ChildItem -LiteralPath 'E:\4DGS\downloads' | Select-Object Name,Length,LastWriteTime
```

Result: stopped only the verified downloader processes. Failed artifacts were preserved rather than deleted.

Clean download command, run in a tracked PTY:

```powershell
curl.exe --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 3 -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' -o 'E:\4DGS\downloads\cut_roasted_beef-model.zip' 'https://api.github.com/repos/juno181/Ex4DGS/releases/assets/202878157'
```

Result: succeeded. Exact size is `109485709` bytes.

### 17. Verify and unpack the official pretrained model

```powershell
$file=Get-Item -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip'
$file | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName
$entries = tar.exe -tf $file.FullName
$entries
Expand-Archive -LiteralPath $file.FullName -DestinationPath 'E:\4DGS\pretrained\cut_roasted_beef' -Force
Get-ChildItem -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef' -Recurse -File | Select-Object FullName,Length | Format-Table -AutoSize
```

Result: archive listing and extraction succeeded. SHA-256 was printed again with explicit formatting:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-model.zip' | Format-List Algorithm,Hash,Path
```

Verified SHA-256: `D3BA00C64323E35F3CD3147C26C66B4C09CD3271B7C464704854DC8A3D2F3B01`.

### 18. Inspect pretrained model metadata

```powershell
Get-Content -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef\cfg_args' -Raw
Get-Content -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef\mean_metrics.json' -Raw
$env:PYTHONIOENCODING='utf-8'
& conda run -n Ex4DGS python -c "...camera and PLY inspection..."
```

Result: `cfg_args` and `mean_metrics.json` were read. Error: `conda` was not on `PATH` in this shell.

Fix and environment lookup:

```powershell
$candidates=@(
'C:\Users\CLEFFFF\.conda\envs\Ex4DGS\python.exe',
'C:\Users\CLEFFFF\miniconda3\envs\Ex4DGS\python.exe',
'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe',
'C:\ProgramData\miniconda3\envs\Ex4DGS\python.exe',
'C:\ProgramData\anaconda3\envs\Ex4DGS\python.exe'
)
$candidates | ForEach-Object { [pscustomobject]@{Path=$_; Exists=Test-Path -LiteralPath $_} }
Get-Command conda.exe -All -ErrorAction SilentlyContinue | Select-Object Source
where.exe conda 2>$null
```

Result: located the environment Python at `C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe`.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import json, pathlib; p=pathlib.Path(r'E:\4DGS\pretrained\cut_roasted_beef'); cams=json.load(open(p/'cameras.json')); print('camera_records',len(cams)); print('keys',sorted(cams[0])); print('image_names_sample',[c.get('img_name') or c.get('image_name') for c in cams[:5]]); print('unique_width_height',sorted({(c['width'],c['height']) for c in cams})); print('first_camera',cams[0]);
for n in ['input.ply','point_cloud/iteration_40000/point_cloud.ply','point_cloud/iteration_40000/dynamic_point_cloud.ply']:
 f=open(p/n,'rb'); h=b''
 while b'end_header\n' not in h: h+=f.readline()
 print('---',n,'---'); print(h.decode('ascii'))"
```

Result: succeeded. Model `cfg_args` records `source_path=.../cut_roasted_beef`, `loader=neural3dvideo`, `resolution=2`, `duration=300`, `start_timestamp=0`, `end_timestamp=-1`, `time_interval=10`, `time_pad=2`, and iteration 40000. `cameras.json` has 6,000 records = 20 cameras × 300 frames at original 2704×2028 metadata resolution. The model contains 185,033 static and 63,862 dynamic Gaussians.

### 19. Download the official N3V cut_roasted_beef scene

Initial release API query:

```powershell
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0' -Headers @{ 'User-Agent'='Codex-Reproduction' }
$asset = $release.assets | Where-Object name -eq 'cut_roasted_beef.zip'
$asset | Select-Object id,name,size,url,browser_download_url,digest | Format-List
```

Result: failed once with unexpected EOF. Retry command:

```powershell
$ErrorActionPreference='Stop'
for($i=1;$i -le 5;$i++) {
  try {
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0' -Headers @{ 'User-Agent'='Codex-Reproduction' }
    break
  } catch {
    Write-Host "Attempt $i failed: $($_.Exception.Message)"
    if($i -lt 5){ Start-Sleep -Seconds 2 }
  }
}
$asset = $release.assets | Where-Object name -eq 'cut_roasted_beef.zip'
$asset | Select-Object id,name,size,url,browser_download_url,digest | Format-List
```

Result: first retry reported SSL failure; the next succeeded. Asset ID is `59818036`, expected size `1143395383`; no upstream digest is published.

Single-connection API download:

```powershell
curl.exe --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 3 -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' -o 'E:\4DGS\downloads\cut_roasted_beef-data.zip' 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59818036'
```

Result: transfer stabilized at approximately 40–60 KB/s with an estimate of 6–7 hours. It was interrupted with Ctrl+C after preserving the partial file.

Redirect diagnostics and 10 MiB range test:

```powershell
curl.exe --http1.1 -sS -D - -o NUL -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59818036'
$resp = Invoke-WebRequest -UseBasicParsing -MaximumRedirection 0 -SkipHttpErrorCheck -Uri 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59818036' -Headers @{ 'Accept'='application/octet-stream'; 'User-Agent'='Codex-Reproduction' }
$url = $resp.Headers.Location
"redirect_host=$(([uri]$url).Host)"
curl.exe --http1.1 -L --fail --range 0-10485759 -o 'E:\4DGS\downloads\n3v-speedtest.part' $url
Get-Item -LiteralPath 'E:\4DGS\downloads\n3v-speedtest.part' | Select-Object Length,LastWriteTime
```

Result: `Invoke-WebRequest` reported that the maximum redirection count was exceeded and `$url` was a string array, but `curl` still received the redirect URL and the 10 MiB range request completed successfully. A subsequent single-connection resume against the signed `release-assets.githubusercontent.com` URL remained too slow and was also interrupted.

Fix: use byte-range downloads against the official signed asset URL. First attempt used 35 chunks of 32 MiB with 12 workers; 15 chunks finished, while 12 connections stalled. The command was interrupted and partial chunks were retained. The full command was:

```powershell
$headers = & curl.exe --http1.1 -sS -D - -o NUL -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59818036'
$locationLine = $headers | Where-Object { $_ -like 'Location:*' } | Select-Object -First 1
$url = ($locationLine -replace '^Location:\s*','').Trim()
$assetSize = 1143395383L
$chunkSize = 33554432L
$partDir = 'E:\4DGS\downloads\n3v-cut-roasted-beef-parts'
New-Item -ItemType Directory -Force -Path $partDir | Out-Null
$partCount = [math]::Ceiling($assetSize / $chunkSize)
0..($partCount-1) | ForEach-Object -Parallel {
  $i = $_
  $start = $i * $using:chunkSize
  $end = [math]::Min($start + $using:chunkSize - 1, $using:assetSize - 1)
  $out = Join-Path $using:partDir ('part-{0:D3}.bin' -f $i)
  & curl.exe --http1.1 -sS -L --fail --retry 10 --retry-all-errors --range "$start-$end" --output $out $using:url
  if($LASTEXITCODE -ne 0) { throw "curl failed for part $i range $start-$end with exit $LASTEXITCODE" }
  $actual = (Get-Item -LiteralPath $out).Length
  $expected = $end - $start + 1
  if($actual -ne $expected) { throw "size mismatch part $i expected=$expected actual=$actual" }
  "completed part=$i bytes=$actual"
} -ThrottleLimit 12
```

Inspection commands confirmed the completed/partial chunks and exact `curl.exe` child processes. A four-worker retry with `--max-time 300` completed two more chunks but left four connections stalled, so it was interrupted. Final fix was to use 20-second connection recycling and 50 retries:

```powershell
$headers = & curl.exe --http1.1 -sS -D - -o NUL -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59818036'
$locationLine = $headers | Where-Object { $_ -like 'Location:*' } | Select-Object -First 1
$url = ($locationLine -replace '^Location:\s*','').Trim()
$assetSize = 1143395383L
$chunkSize = 33554432L
$partDir = 'E:\4DGS\downloads\n3v-cut-roasted-beef-parts'
$partCount = [math]::Ceiling($assetSize / $chunkSize)
$missing = 0..($partCount-1) | Where-Object {
  $start = $_ * $chunkSize
  $end = [math]::Min($start + $chunkSize - 1, $assetSize - 1)
  $path = Join-Path $partDir ('part-{0:D3}.bin' -f $_)
  (-not (Test-Path -LiteralPath $path)) -or ((Get-Item -LiteralPath $path).Length -ne ($end-$start+1))
}
$missing | ForEach-Object -Parallel {
  $i = $_
  $start = $i * $using:chunkSize
  $end = [math]::Min($start + $using:chunkSize - 1, $using:assetSize - 1)
  $out = Join-Path $using:partDir ('part-{0:D3}.bin' -f $i)
  & curl.exe --http1.1 -sS -L --fail --connect-timeout 10 --max-time 20 --retry 50 --retry-all-errors --retry-delay 0 --range "$start-$end" --output $out $using:url
  if($LASTEXITCODE -ne 0) { throw "curl failed for part $i range $start-$end with exit $LASTEXITCODE" }
  $actual = (Get-Item -LiteralPath $out).Length
  $expected = $end - $start + 1
  if($actual -ne $expected) { throw "size mismatch part $i expected=$expected actual=$actual" }
  "completed fast-retry part=$i bytes=$actual"
} -ThrottleLimit 8
Get-ChildItem -LiteralPath $partDir -File | Measure-Object -Property Length -Sum | Select-Object Count,Sum
```

Result: several attempts timed out after 20 seconds and were automatically retried. All 35 chunks ultimately passed exact range-length checks; total size `1143395383` bytes.

Assembly and verification command:

```powershell
$partDir='E:\4DGS\downloads\n3v-cut-roasted-beef-parts'
$outPath='E:\4DGS\downloads\cut_roasted_beef-data-complete.zip'
$parts=Get-ChildItem -LiteralPath $partDir -Filter 'part-*.bin' -File | Sort-Object Name
$outStream=[System.IO.File]::Open($outPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
try {
  foreach($part in $parts) {
    $inStream=[System.IO.File]::OpenRead($part.FullName)
    try { $inStream.CopyTo($outStream) } finally { $inStream.Dispose() }
  }
} finally { $outStream.Dispose() }
Get-Item -LiteralPath $outPath | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $outPath | Format-List Algorithm,Hash,Path
tar.exe -tf $outPath
```

Result: succeeded. The assembled archive has the exact release size, lists cleanly as a ZIP, and has local SHA-256 `2F9D3BD8782B96425E8ADB506FA4CBDFAAB4AB753A8FDAF44B9B2468000B8C4F`. It contains 20 camera videos (`cam00` through `cam20`, excluding invalid `cam04`) and `poses_bounds.npy`.

### 20. Extract and inventory the N3V scene

```powershell
Expand-Archive -LiteralPath 'E:\4DGS\downloads\cut_roasted_beef-data-complete.zip' -DestinationPath 'E:\4DGS\data' -Force
Get-ChildItem -LiteralPath 'E:\4DGS\data\cut_roasted_beef' -File | Sort-Object Name | Select-Object Name,Length,LastWriteTime
(Get-ChildItem -LiteralPath 'E:\4DGS\data\cut_roasted_beef' -Filter 'cam*.mp4' -File).Count
Get-Item -LiteralPath 'E:\4DGS\data\cut_roasted_beef\poses_bounds.npy' | Select-Object FullName,Length,LastWriteTime
```

Result: succeeded. Extracted 20 MP4 files plus `poses_bounds.npy` into `E:\4DGS\data\cut_roasted_beef`.

### 21. Check preprocessing dependencies

```powershell
Get-Command colmap.exe,ffmpeg.exe -All -ErrorAction SilentlyContinue | Select-Object Name,Source
& 'C:\Users\CLEFFFF\anaconda3\Scripts\conda.exe' env list
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import sys, cv2, torch, numpy; print('python',sys.version); print('opencv',cv2.__version__); print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available()); print('numpy',numpy.__version__)"
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -m pip check
```

Result: Ex4DGS environment is healthy: Python 3.9.25, OpenCV 4.11.0, PyTorch 2.1.2+CUDA 11.8, CUDA available, NumPy 1.26.4, and no broken requirements. Neither COLMAP nor FFmpeg is currently on `PATH`; no `colmapenv` exists.

```powershell
Get-Content -LiteralPath 'E:\4DGS\scripts\env_setup.sh' -Raw
Get-Content -LiteralPath 'E:\4DGS\requirements.txt' -Raw
& 'C:\Users\CLEFFFF\anaconda3\Scripts\conda.exe' search -c conda-forge colmap --platform win-64
```

Result: the repository preprocessing setup script creates a separate `colmapenv` and installs COLMAP from conda-forge. The package search encountered repeated `SSLEOFError` responses while fetching Conda metadata and did not return a package result within the foreground window.

### 22. Validate raw scene video, pose, camera, and time metadata

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'conda.*search.*colmap' } | Select-Object ProcessId,Name,CreationDate,CommandLine | Format-List
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import cv2, glob, os, numpy as np; root=r'E:\4DGS\data\cut_roasted_beef'; vids=sorted(glob.glob(os.path.join(root,'cam*.mp4'))); print('video_count',len(vids));
for v in vids:
 c=cv2.VideoCapture(v); print(os.path.basename(v), 'frames',int(c.get(cv2.CAP_PROP_FRAME_COUNT)),'size',int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),'fps',c.get(cv2.CAP_PROP_FPS),'opened',c.isOpened()); c.release()
p=np.load(os.path.join(root,'poses_bounds.npy')); print('poses_bounds_shape',p.shape,'dtype',p.dtype); poses=p[:,:15].reshape(-1,3,5); print('pose_count',len(poses)); print('unique_HWF',sorted({tuple(map(float,x[:,-1])) for x in poses})); print('bounds_minmax',float(p[:,-2:].min()),float(p[:,-2:].max()))"
```

Result: all 20 videos open successfully; every video is exactly 300 frames, 2704×2028, and 30 FPS. `poses_bounds.npy` is shape `(20,17)` and contains 20 matching poses with focal length `1462.7383255729974`. Missing camera ID `cam04` is consistent with the official N3V note that invalid streams were removed.

The repository conversion and camera serialization functions were inspected:

```powershell
rg -n -A 45 -B 10 "def camera_to_JSON|def posetow2c_matrcs" 'E:\4DGS\scene\cameras.py' 'E:\4DGS\utils\camera_utils.py' 'E:\4DGS\dataset_utils\etc_utils.py' 'E:\4DGS\utils'
```

Then the model's first held-out camera was compared numerically with the raw scene pose:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import json,numpy as np,sys,pathlib; sys.path.insert(0,r'E:\4DGS'); from dataset_utils.etc_utils import posetow2c_matrcs; d=pathlib.Path(r'E:\4DGS\data\cut_roasted_beef'); p=np.load(d/'poses_bounds.npy'); poses=p[:,:15].reshape(-1,3,5); ms=posetow2c_matrcs(poses.transpose(1,2,0)); c2w=np.linalg.inv(ms[0]); cams=json.load(open(r'E:\4DGS\pretrained\cut_roasted_beef\cameras.json')); c=cams[0]; pos=np.array(c['position']); rot=np.array(c['rotation']); print('raw_cam0_expected_position',c2w[:3,3]); print('model_cam0_position',pos); print('max_abs_position_diff',np.max(np.abs(pos-c2w[:3,3]))); print('max_abs_rotation_diff',np.max(np.abs(rot-c2w[:3,:3]))); print('model_fx_fy',c['fx'],c['fy']); print('raw_focal',poses[0,2,4]); print('focal_abs_diff',abs(c['fx']-poses[0,2,4])); print('model_wh',(c['width'],c['height']),'raw_wh',(poses[0,1,4],poses[0,0,4]))"
```

Result: scene/model match is numerically exact within floating-point precision: maximum camera-position difference `1.1102230246251565e-16`, maximum rotation difference `6.938893903907228e-17`, focal difference `2.2737367544323206e-13`, and identical 2704×2028 dimensions.

### 23. Investigate official COLMAP Windows binary as a preprocessing fallback

Action: checked official COLMAP documentation and releases. Official documentation states that prebuilt Windows binaries contain the CLI and can be run through `COLMAP.bat`.

```powershell
$ErrorActionPreference='Stop'
for($i=1;$i -le 5;$i++) { try { $rel=Invoke-RestMethod -Uri 'https://api.github.com/repos/colmap/colmap/releases/tags/4.1.0' -Headers @{'User-Agent'='Codex-Reproduction'}; break } catch { "attempt $i failed: $($_.Exception.Message)"; if($i -lt 5){Start-Sleep -Seconds 2} } }
$rel | Select-Object tag_name,name,published_at,html_url
$rel.assets | Select-Object id,name,size,browser_download_url,digest | Format-Table -AutoSize
$rel.assets | ForEach-Object { $_ | Select-Object id,name,size,url,browser_download_url,digest | Format-List }
```

Result: located official COLMAP 4.1.0 Windows no-CUDA asset ID `458634077`, size `118876963`, published SHA-256 `dc8179bb4f3f48edec683bcec7627176b66e53a33ef0e2aa98d487f45873af5f`.

```powershell
curl.exe --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 2 -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-Reproduction' -o 'E:\4DGS\downloads\colmap-x64-windows-nocuda-4.1.0.zip' 'https://api.github.com/repos/colmap/colmap/releases/assets/458634077'
```

Result: the GitHub asset connection again became persistently slow. It was interrupted at about 46 MiB instead of waiting more than ten additional minutes. The partial file was retained. COLMAP was not needed for inference because the raw N3V poses were proven numerically identical to the pretrained model's camera metadata and the model archive includes the exact initialization point cloud.

### 24. Work around the repository frame-extraction bug without changing source code

Inspection found that `scripts/pre_n3d_colmap.py::extractframes` uses:

```python
if all(os.path.join(output_dir, "{:06d}.png".format(i)) for i in range(startframe, endframe)):
```

This tests non-empty path strings rather than file existence, so it always skips extraction. No tracked source file was edited. For pretrained test-view rendering, all 300 `cam00` frames and frame 0 of the other 19 cameras were extracted. One frame per training camera is sufficient to compute the identical camera normalization; training views are skipped during evaluation.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import cv2,glob,os,pathlib,sys; root=pathlib.Path(r'E:\4DGS\data\cut_roasted_beef'); vids=sorted(root.glob('cam*.mp4')); total=0
for v in vids:
 out=v.with_suffix(''); out.mkdir(exist_ok=True); cap=cv2.VideoCapture(str(v)); want=300 if v.stem=='cam00' else 1; n=0
 while n<want:
  ok,frame=cap.read()
  if not ok: raise RuntimeError(f'failed reading {v} frame {n}')
  dst=out/f'{n:06d}.png'
  if not cv2.imwrite(str(dst),frame): raise RuntimeError(f'failed writing {dst}')
  n+=1; total+=1
 cap.release(); print(v.name,'extracted',n)
print('total_extracted',total)"
```

Result: succeeded. Follow-up counts showed 300 PNGs in `cam00`, one PNG in each of the other 19 camera directories, 319 PNGs total.

### 25. Create inference-only COLMAP text metadata from official poses

Created helper `C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\build_n3v_render_metadata.py` with `apply_patch`. The helper calls the repository's unchanged `posetow2c_matrcs` and `rotmat2qvec`, writes standard COLMAP `cameras.txt`/`images.txt`, and copies the official model archive's `input.ply` to `points3D.ply`. It is outside the repository and does not change algorithm code.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\build_n3v_render_metadata.py'
Get-Content -LiteralPath 'E:\4DGS\data\cut_roasted_beef\colmap_0\sparse\0\cameras.txt'
Get-Content -LiteralPath 'E:\4DGS\data\cut_roasted_beef\colmap_0\sparse\0\images.txt' -TotalCount 8
Get-ChildItem -LiteralPath 'E:\4DGS\data\cut_roasted_beef\colmap_0\sparse\0' -File | Select-Object Name,Length,LastWriteTime
```

Result: wrote 20 PINHOLE cameras and 20 image poses; copied the official 6,141-point initialization cloud.

Loader preflight:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import sys; from types import SimpleNamespace; sys.path.insert(0,r'E:\4DGS'); from scene.dataset_readers import readColmapSceneInfoNeural3DVideo; a=SimpleNamespace(start_timestamp=0,end_timestamp=-1); s=readColmapSceneInfoNeural3DVideo(r'E:\4DGS\data\cut_roasted_beef','images',True,a); print('train_records',len(s.train_cameras)); print('test_records',len(s.test_cameras)); print('train_unique_camera_ids',len({c.uid for c in s.train_cameras}),sorted({c.uid for c in s.train_cameras})); print('test_unique_camera_ids',len({c.uid for c in s.test_cameras}),sorted({c.uid for c in s.test_cameras})); print('train_unique_times',len({c.timestamp for c in s.train_cameras}),min(c.timestamp for c in s.train_cameras),max(c.timestamp for c in s.train_cameras)); print('test_unique_times',len({c.timestamp for c in s.test_cameras}),min(c.timestamp for c in s.test_cameras),max(c.timestamp for c in s.test_cameras)); print('train_paths_exist',all(__import__('os').path.exists(c.image_path) for c in s.train_cameras)); print('test_paths_exist',all(__import__('os').path.exists(c.image_path) for c in s.test_cameras)); print('normalization_radius',s.nerf_normalization['radius']); print('point_count',len(s.point_cloud.points))"
```

Result: passed. Test split is one held-out camera (ID 1 / raw `cam00`) with timestamps 0–299 and 300 existing frames. Training metadata has the other 19 cameras. Point cloud count is 6,141.

### 26. Preserve the official model and verify algorithm sources are unchanged

```powershell
git status --short
git diff --stat
git diff --name-only
Get-ChildItem -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef' -Recurse -File | Measure-Object -Property Length -Sum | Select-Object Count,Sum
New-Item -ItemType Directory -Force -Path 'E:\4DGS\runs\cut_roasted_beef' | Out-Null
Copy-Item -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef\*' -Destination 'E:\4DGS\runs\cut_roasted_beef' -Recurse -Force
```

Result: tracked diff was empty. Error: `Copy-Item -LiteralPath` does not expand `*`, so no files were copied.

Fix:

```powershell
Copy-Item -Path 'E:\4DGS\pretrained\cut_roasted_beef\*' -Destination 'E:\4DGS\runs\cut_roasted_beef' -Recurse -Force
Get-ChildItem -LiteralPath 'E:\4DGS\runs\cut_roasted_beef' -Recurse -File | Measure-Object -Property Length -Sum | Select-Object Count,Sum
```

Result: succeeded; copied seven model files totaling 131,219,602 bytes to the run directory.

### 27. Attempt the repository's official evaluation entry point

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' 'E:\4DGS\render.py' --model_path 'E:\4DGS\runs\cut_roasted_beef' --source_path 'E:\4DGS\data\cut_roasted_beef' --skip_train --iteration 40000 --save_img
```

Result: model iteration 40000 loaded successfully, training/test cameras loaded successfully, and the console printed `300 300`, confirming model and dataset duration agreement. Before rendering frame 0, evaluation attempted to download torchvision AlexNet weights (`alexnet-owt-7be5be79.pth`, 233 MiB) for LPIPS. The download stalled at approximately 57.2 MiB for several minutes.

Diagnostics:

```powershell
Get-ChildItem -LiteralPath 'C:\Users\CLEFFFF\.cache\torch\hub\checkpoints' -Force -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'render.py|alexnet-owt' } | Select-Object ProcessId,Name,CreationDate,CommandLine | Format-List
```

Fix: after PTY Ctrl+C did not terminate the blocked URL-retrieval child cleanly, the exact verified render wrapper and Python PIDs were stopped:

```powershell
Stop-Process -Id 11000,36232 -Force
Start-Sleep -Seconds 1
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'E:\\4DGS\\render.py' } | Select-Object ProcessId,Name,CommandLine
```

Result: official evaluator was stopped. This is an LPIPS dependency-download failure, not a Gaussian model/camera/render initialization failure.

### 28. Render the complete held-out scene without changing algorithm code

Inspected the camera and metric utilities:

```powershell
rg -n -A 90 -B 20 "def loadCamVideo|class Camera" 'E:\4DGS\scene\cameras.py'
Get-Content -LiteralPath 'E:\4DGS\utils\image_utils.py' -Raw
rg -n -A 35 -B 5 "def PILtoTorch" 'E:\4DGS\utils\general_utils.py'
```

Created orchestration helper `C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\render_pretrained_scene.py` with `apply_patch`. It imports and calls the repository's unchanged `Scene`, `CGaussianModel`, and `gaussian_renderer.render`, renders all test frames, saves PNGs, and computes the repository's PSNR and SSIM metrics. It omits only LPIPS, whose unrelated ImageNet backbone download was blocked.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\render_pretrained_scene.py'
```

Result: succeeded. All 300 held-out `cam00` frames at timestamps 0–299 rendered at 1352×1014. Recorded results:

- Mean PSNR: `33.7313449605306` dB.
- Mean repository SSIM metric: `0.9576468815406164`.
- Mean synchronized core render time: `0.037036764333332875` seconds/frame.
- Mean synchronized core render rate: `27.00019880246412` FPS.

The synchronized core FPS is not directly comparable to the paper's timing protocol or the repository's unsynchronized benchmark loop.

### 29. Inspect and package rendering outputs

```powershell
Get-Content -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\reproduction_render_metrics.json' -Raw
$renderDir='E:\4DGS\runs\cut_roasted_beef\test\itrs_40000\renders'
$gtDir='E:\4DGS\runs\cut_roasted_beef\test\itrs_40000\gt_samples'
Get-ChildItem -LiteralPath $renderDir -Filter '*.png' -File | Measure-Object -Property Length -Sum | Select-Object Count,Sum
Get-ChildItem -LiteralPath $gtDir -Filter '*.png' -File | Sort-Object Name | Select-Object Name,Length,LastWriteTime
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "from PIL import Image; from pathlib import Path; r=Path(r'E:\4DGS\runs\cut_roasted_beef\test\itrs_40000\renders');
for n in ['000000.png','000150.png','000299.png']:
 im=Image.open(r/n); print(n,im.size,im.mode)"
```

Result: 300 render PNGs totaling 250,975,106 bytes; sample images are RGB 1352×1014. Visual inspection of render/ground-truth pairs at frames 0, 150, and 299 confirmed the same scene, held-out viewpoint, and chronological motion. Expected reconstruction blur/ghosting is visible on moving objects, but no camera or timestamp mismatch is present.

Video encoding and verification:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import cv2,glob,os; src=r'E:\4DGS\runs\cut_roasted_beef\test\itrs_40000\renders'; files=sorted(glob.glob(os.path.join(src,'*.png'))); first=cv2.imread(files[0]); h,w=first.shape[:2]; out=r'E:\4DGS\runs\cut_roasted_beef\cut_roasted_beef_cam00_iter40000.mp4'; writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),30.0,(w,h));
if not writer.isOpened(): raise RuntimeError('VideoWriter failed to open')
for f in files:
 im=cv2.imread(f)
 if im is None or im.shape[:2]!=(h,w): raise RuntimeError(f'bad frame {f}')
 writer.write(im)
writer.release(); cap=cv2.VideoCapture(out); print('output',out); print('frames',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'size',int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'fps',cap.get(cv2.CAP_PROP_FPS),'opened',cap.isOpened()); cap.release()"
Get-Item -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\cut_roasted_beef_cam00_iter40000.mp4' | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\cut_roasted_beef_cam00_iter40000.mp4' | Format-List Algorithm,Hash,Path
```

Result: created a verified 300-frame, 1352×1014, 30 FPS MP4. Size `3,411,243` bytes; SHA-256 `772C7293CE1CEE8DB431353461C1120CA9207A808142F347D61742E161D9E56F`.

### 30. Final integrity, metric, source-state, GPU, and disk checks

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import json,glob,os; from PIL import Image; run=r'E:\4DGS\runs\cut_roasted_beef'; files=sorted(glob.glob(os.path.join(run,'test','itrs_40000','renders','*.png'))); sizes=set(); modes=set();
for f in files:
 with Image.open(f) as im: im.verify()
 with Image.open(f) as im: sizes.add(im.size); modes.add(im.mode)
rep=json.load(open(os.path.join(run,'reproduction_render_metrics.json'))); off=json.load(open(r'E:\4DGS\pretrained\cut_roasted_beef\mean_metrics.json')); print('verified_png_count',len(files)); print('unique_sizes',sorted(sizes)); print('modes',sorted(modes)); print('reproduced_psnr',rep['psnr_mean']); print('official_archive_psnr',off['PSNR']); print('abs_psnr_difference',abs(rep['psnr_mean']-off['PSNR'])); print('reproduced_ssim_code_metric',rep['ssim_mean']); print('official_archive_ssim_code_metric',off['SSIM']); print('abs_ssim_difference',abs(rep['ssim_mean']-off['SSIM']))"
git rev-parse HEAD
git remote -v
git diff --exit-code
git ls-files -m
git status --short
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Get-PSDrive -Name E | Select-Object Name,Used,Free
```

Result:

- All 300 PNGs decode successfully and are uniformly RGB 1352×1014.
- Reproduced PSNR differs from the official model archive by `0.001557566324869697` dB; both round to the paper's `33.73` dB for Cut Roasted Beef.
- Reproduced repository SSIM differs from the archive's stored repository SSIM by `7.952253023746358e-06`. This repository SSIM is not the paper's separately reported scikit-image SSIM1/SSIM2 value.
- Git commit: `1e4539eea2d2a15d286db003e0c4ae8cca4ec792`.
- `git diff --exit-code` succeeded and `git ls-files -m` returned nothing: no tracked algorithm source changed.
- Only untracked experiment artifacts are present: `EXPERIMENT_LOG.md`, `Papers/`, `data/`, `pretrained/`, and `runs/`.
- GPU: NVIDIA GeForce RTX 4090, driver 616.92, 24,564 MiB.
- E: free space after the run: 1,414,103,298,048 bytes.

### 31. Diagnose the missing AlexNet dependency

The official PyTorch URL and the stale partial download left by the interrupted repository evaluation were inspected:

```powershell
curl.exe --http1.1 -I --retry 5 --retry-all-errors --connect-timeout 15 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
Get-ChildItem -LiteralPath 'C:\Users\CLEFFFF\.cache\torch\hub\checkpoints' -Force -Filter 'alexnet*' -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime
```

Result: HTTP 200, `Content-Length: 244408911`, `Accept-Ranges: bytes`, and ETag `"30798c2b5ec677eae8ec897cf56cdba6-15"`. The old partial file `alexnet-owt-7be5be79.pth.08717482ce4a4db09c3aac643d5af6e6.partial` was only 59,998,208 bytes. It was retained unchanged for traceability.

The local LPIPS implementation and available transfer tools were then inspected:

```powershell
Get-ChildItem -LiteralPath 'E:\4DGS\lpipsPyTorch' -Recurse -File | Select-Object FullName,Length
rg -n "def lpips|class LPIPS|alex" 'E:\4DGS\lpipsPyTorch'
Get-Content -LiteralPath 'E:\4DGS\lpipsPyTorch\__init__.py'
Get-Content -LiteralPath 'E:\4DGS\lpipsPyTorch\modules\lpips.py'
Get-Content -LiteralPath 'E:\4DGS\lpipsPyTorch\modules\utils.py'
Get-Content -LiteralPath 'E:\4DGS\lpipsPyTorch\modules\networks.py'
$PSVersionTable.PSVersion.ToString()
Get-Command aria2c -ErrorAction SilentlyContinue | Select-Object Source
Get-Command curl.exe | Select-Object Source
```

Result: the repository uses torchvision's ImageNet AlexNet feature extractor plus LPIPS v0.1 linear calibration weights. Its `lpips(...)` convenience function constructs and loads the full criterion on every call. PowerShell 7.6.5 and `curl.exe` were available; `aria2c` was not installed.

### 32. Download, assemble, hash-check, and cache AlexNet

Created the external helper `C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\download_alexnet.ps1` with `apply_patch`. It splits the declared 244,408,911-byte object into 16 MiB HTTP ranges, downloads up to eight ranges concurrently, validates every part length, assembles the parts in byte order, verifies the SHA-256 filename prefix, and copies the verified result into Torch's cache. No repository source was edited.

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\download_alexnet.ps1'
```

Result: all 15 ranges downloaded successfully. The assembled file is `E:\4DGS\downloads\alexnet-owt-7be5be79.pth`, size 244,408,911 bytes. It was copied to `C:\Users\CLEFFFF\.cache\torch\hub\checkpoints\alexnet-owt-7be5be79.pth`. SHA-256:

```text
7BE5BE791159472B1FBF3C69796F7CB30DCA7AD8466C2DF70058C37116CDEE02
```

This matches the expected `7be5be79` hash prefix embedded in the official PyTorch filename.

Offline deserialization and torchvision model construction were verified:

```powershell
$env:TORCH_HOME='C:\Users\CLEFFFF\.cache\torch'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import torch; from torchvision import models; p=r'C:\Users\CLEFFFF\.cache\torch\hub\checkpoints\alexnet-owt-7be5be79.pth'; s=torch.load(p,map_location='cpu'); print('keys',len(s),'first',next(iter(s)),'last',next(reversed(s))); m=models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1); print('loaded',sum(x.numel() for x in m.parameters()))"
Get-ChildItem -LiteralPath 'C:\Users\CLEFFFF\.cache\torch\hub\checkpoints' -Force | Select-Object Name,Length,LastWriteTime
Get-ChildItem -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\test\itrs_40000\renders' -Filter '*.png' | Sort-Object Name | Select-Object -First 3 Name,Length
Get-ChildItem -LiteralPath 'E:\4DGS\data\cut_roasted_beef\cam00' -Filter '*.png' | Sort-Object Name | Select-Object -First 3 Name,Length
```

Result: the state dictionary contains 16 tensors from `features.0.weight` through `classifier.6.bias`; torchvision loaded all 61,100,840 AlexNet parameters from the completed cache file without another backbone download.

### 33. Run full 300-frame AlexNet-LPIPS validation

The dataset image preprocessing and evaluation call path were rechecked before running:

```powershell
rg -n "PILtoTorch|reproduction_render_metrics|cut_roasted" 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work' 'E:\4DGS\render.py' 'E:\4DGS\utils\general_utils.py'
Get-Content -LiteralPath 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\render_pretrained_scene.py'
Get-Content -LiteralPath 'E:\4DGS\render.py' -TotalCount 180
Get-Content -LiteralPath 'E:\4DGS\utils\general_utils.py' -TotalCount 60
rg -n "getTestCameras|cam00|PILtoTorch|resolution_scale|original_image|image_name" 'E:\4DGS\scene' 'E:\4DGS\utils'
Get-Content -LiteralPath 'E:\4DGS\scene\__init__.py' | Select-Object -Skip 180 -First 85
Get-Content -LiteralPath 'E:\4DGS\scene\cameras.py' | Select-Object -Skip 225 -First 80
Get-Content -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\cfg_args'
```

Created external helper `C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\verify_alex_lpips.py` with `apply_patch`. It loads the unchanged repository renderer, official iteration-40000 Gaussian model, and the repository's LPIPS class. It uses `Scene.getTestCameras(...)`, so the original bilinear half-resolution GT preprocessing, test split, camera ordering, timestamps, and image scale are retained exactly. It rerenders floating-point images instead of measuring quantized saved PNGs. The only orchestration optimization is to reuse one identical AlexNet-LPIPS criterion for all 300 frames instead of reconstructing it for every frame.

```powershell
$env:TORCH_HOME='C:\Users\CLEFFFF\.cache\torch'
$env:PYTHONUNBUFFERED='1'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\verify_alex_lpips.py'
```

Result: the first run downloaded the official LPIPS v0.1 AlexNet linear calibration file, `alex.pth` (6,009 bytes), from the URL already specified by repository code. Two torchvision deprecation warnings were printed because repository code calls `models.alexnet(True)`; these are warnings, not errors, and no source change was made. All 300 frames completed successfully.

Measured result:

- Reproduced AlexNet-LPIPS mean: `0.040437700934708116`.
- Official pretrained archive mean: `0.04044107347726822`.
- Absolute difference: `0.0000033725425601024983`.
- Scene: `cut_roasted_beef`.
- Test camera IDs: `[1]`, corresponding to held-out raw `cam00`.
- Timestamp range: 0–299.
- Resolution: 1352×1014.

The full per-frame output was written to `E:\4DGS\runs\cut_roasted_beef\reproduction_lpips_alex.json`.

### 34. LPIPS artifact and source-integrity checks

```powershell
Get-Item -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\reproduction_lpips_alex.json' | Select-Object FullName,Length,LastWriteTime
Get-Content -LiteralPath 'E:\4DGS\runs\cut_roasted_beef\reproduction_lpips_alex.json' -TotalCount 24
Get-Content -LiteralPath 'E:\4DGS\pretrained\cut_roasted_beef\mean_metrics.json'
git -C 'E:\4DGS' status --short
git -C 'E:\4DGS' diff --exit-code
```

Then the cache hashes and result aggregation were independently checked:

```powershell
$alex='C:\Users\CLEFFFF\.cache\torch\hub\checkpoints\alexnet-owt-7be5be79.pth'
$lin='C:\Users\CLEFFFF\.cache\torch\hub\checkpoints\alex.pth'
Get-Item -LiteralPath $alex,$lin | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $alex,$lin | Format-List Algorithm,Hash,Path
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import json,pathlib; p=pathlib.Path(r'E:\4DGS\runs\cut_roasted_beef\reproduction_lpips_alex.json'); d=json.loads(p.read_text()); vals=d['per_frame']; print('frame_count',len(vals)); print('first_last',next(iter(vals)),next(reversed(vals))); print('recomputed_mean',sum(vals.values())/len(vals)); print('stored_mean',d['lpips_alex_mean']); print('official',d['official_lpips_alex_mean']); print('abs_diff',d['absolute_difference']); assert len(vals)==300 and abs(sum(vals.values())/len(vals)-d['lpips_alex_mean'])<1e-15"
git -C 'E:\4DGS' diff --exit-code
git -C 'E:\4DGS' ls-files -m
```

Result:

- The output JSON contains exactly 300 entries, from `000000.png` through `000299.png`.
- Independently recomputing the arithmetic mean from all per-frame values exactly reproduces `0.040437700934708116`.
- AlexNet backbone SHA-256: `7BE5BE791159472B1FBF3C69796F7CB30DCA7AD8466C2DF70058C37116CDEE02`.
- LPIPS v0.1 AlexNet calibration SHA-256: `DF73285E35B22355A2DF87CDB6B70B343713B667EDDBDA73E1977E0C860835C0`.
- `git diff --exit-code` succeeded and `git ls-files -m` returned nothing. No tracked algorithm source was modified.

### 35. Inventory the remaining official N3V release assets

The repository scene list, existing data, download artifacts, and free space were inspected:

```powershell
rg -n -i "n3v|neural 3d video|coffee|flame|cook|spinach|beef|sear|dataset|download" README.md configs scripts EXPERIMENT_LOG.md
Get-ChildItem -LiteralPath 'E:\4DGS\data' -Force | Select-Object Name,Mode,Length,LastWriteTime
Get-ChildItem -LiteralPath 'E:\4DGS\downloads' -Force | Select-Object Name,Mode,Length,LastWriteTime
Get-PSDrive -Name E | Select-Object Name,Used,Free
```

The repository's `scripts/preprocess_all_n3v.sh` lists six N3V scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, and `sear_steak`. Only `cut_roasted_beef` was already present.

Official GitHub release metadata was fetched again:

```powershell
$ErrorActionPreference='Stop'
$rel=Invoke-RestMethod -Uri 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/tags/v1.0' -Headers @{'User-Agent'='Codex-N3V-Download'}
$rel.assets | Sort-Object name | Select-Object id,name,size,digest,browser_download_url | Format-Table -AutoSize
$total=($rel.assets | Where-Object name -ne 'cut_roasted_beef.zip' | Measure-Object size -Sum).Sum
"remaining_asset_bytes=$total"
"remaining_asset_gib=$([math]::Round($total/1GB,3))"
Get-PSDrive -Name E | Format-List Name,Used,Free
```

Result: eight remaining release assets total 9,778,339,298 bytes (9.107 GiB). `flame_salmon_1` is a four-volume ZIP (`.z01`, `.z02`, `.z03`, `.zip`). E: had 1,413,605,961,728 bytes free.

Available transfer and archive tools were checked:

```powershell
Get-Command 7z,7za,tar -ErrorAction SilentlyContinue | Select-Object Name,Source,Version
Get-ChildItem -LiteralPath 'C:\Program Files','C:\Program Files (x86)' -Filter '7z.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 5 FullName
curl.exe --version | Select-Object -First 2
```

Result: Windows `tar.exe` and curl 8.19.0 were available; 7-Zip was not installed.

### 36. Establish a reliable official-asset range download

A direct GitHub release URL range preflight was attempted:

```powershell
$u = & curl.exe --http1.1 --silent --show-error --location --head --output NUL --write-out '%{url_effective}' 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip'
& curl.exe --http1.1 --fail --location --silent --show-error --range 0-1 --output 'E:\4DGS\downloads\n3v-range-preflight.bin' 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip'
```

Error: the direct connection reset with curl error 56.

Fix: resolve the temporary official release-asset URL through the GitHub API, then range-request the signed asset URL:

```powershell
$headers = & curl.exe --http1.1 --silent --show-error --dump-header - --output NUL -H 'Accept: application/octet-stream' -H 'User-Agent: Codex-N3V-Download' 'https://api.github.com/repos/facebookresearch/Neural_3D_Video/releases/assets/59817219'
$loc = (($headers | Where-Object { $_ -match '^location:' }) -replace '^location:\s*','').Trim()
& curl.exe --http1.1 --fail --silent --show-error --range 0-1 --connect-timeout 15 --max-time 60 --output 'E:\4DGS\downloads\n3v-range-preflight.bin' $loc
Get-Item -LiteralPath 'E:\4DGS\downloads\n3v-range-preflight.bin' | Select-Object Length,LastWriteTime
```

Result: succeeded with exactly two bytes, confirming byte-range support.

Created `C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\download_remaining_n3v.ps1` with `apply_patch`. The helper resolves each official asset ID, downloads independently verifiable ranges, assembles only complete ranges, checks the declared release size, computes SHA-256, and writes `E:\4DGS\downloads\n3v-all\download_manifest.json`.

Initial invocation:

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\download_remaining_n3v.ps1'
```

Error: one 32 MiB `coffee_martini` range timed out after 120 seconds with 2,653,435 bytes. PowerShell 7 promoted curl's nonzero exit into a terminating native-command error before the scripted retry loop could continue.

Fix attempts and diagnostics:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'curl.exe'" | Where-Object { $_.CommandLine -match 'release-assets.*coffee_martini|n3v-all-parts' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine
```

The helper was patched to launch curl through hidden `Start-Process -Wait -PassThru`, inspect its exit code, and retain completed parts. A first string interpolation edit produced this parser error:

```text
Variable reference is not valid. ':' was not followed by a valid variable name character.
```

It was fixed by changing `$tempPath:` to `${tempPath}:`.

The last `coffee_martini` 32 MiB range remained abnormally slow. HTTP `--append` did not provide valid HTTP range continuation and was removed. Created `complete_coffee_slow_part.ps1` with `apply_patch`; it divided only that exact range into sixteen 2 MiB ranges, downloaded them concurrently, validated them, and reconstructed the original 33,554,432-byte part:

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\complete_coffee_slow_part.ps1'
```

Result: succeeded. The main downloader then assembled `coffee_martini.zip`, size 1,186,324,684 bytes, SHA-256 `CBC31291CE143E6F31A00F23A049663A61CC8793AF8DAB259DBC6E338A676816`.

### 37. Transfer-strategy diagnostics and final 2 MiB downloader

Windows BITS was tested as a resumable whole-file alternative:

```powershell
Get-Command Start-BitsTransfer,Get-BitsTransfer,Complete-BitsTransfer -ErrorAction SilentlyContinue
$job=Start-BitsTransfer -Source 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip' -Destination 'E:\4DGS\downloads\n3v-all\cook_spinach.bits.zip' -DisplayName 'N3V cook_spinach' -Description 'Official N3V v1.0 asset' -Asynchronous
Get-BitsTransfer -Name 'N3V cook_spinach' | Select-Object JobId,JobState,BytesTransferred,BytesTotal,ErrorDescription
```

Seven N3V BITS jobs were created for a throughput comparison. `Get-BitsTransfer -AllUsers` printed access-denied errors (`0x80070005`) because elevation was not available, although the current user's jobs were created and visible through `bitsadmin /list /verbose`. Aggregate throughput was substantially below the validated range approach.

Fix: cancel only the seven named experimental N3V BITS jobs; their temporary/preallocated files were discarded, while validated range parts were preserved:

```powershell
$ids=@('{E5E38F2F-8F79-4ABB-9FDF-59C3A3D07B6E}','{3E046618-AB2A-443A-B1C6-489BB15D075C}','{D43ED388-3355-4FA3-8E94-EF35F77A306D}','{748A1B02-9DF9-4E18-BC81-BCB09BD3495B}','{1E223D22-C15B-48FD-87A6-CE961AD0E385}','{25262FD5-E0AC-4B70-9F31-BF365D6E2851}','{E29A7713-2773-4B7A-B5EA-5BBC1B551326}')
foreach($id in $ids){bitsadmin /cancel $id}
```

The main downloader was adjusted to 2 MiB ranges, 32 workers, a 60-second per-range timeout, and up to 20 retry rounds. Valid `cook_spinach` 8 MiB parts were preserved and mechanically split into 2 MiB parts rather than downloaded again. The conversion helper was created with `apply_patch` and run:

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\migrate_cook_parts_to_2mb.ps1'
```

Result: reused 108 parts totaling 226,492,416 bytes. The official download was resumed repeatedly with the same idempotent command:

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\download_remaining_n3v.ps1'
```

Expected transient output included curl timeout 28 and reset 56 errors. Any range whose byte count differed from the exact expected count was rejected and retried.

### 38. Handle two reproducible CDN range-boundary cases

For `flame_salmon_1_split.z03`, range `301989888-304087039` reproducibly alternated between two complementary response lengths, 2,094,396 and 2,756 bytes, rather than returning the requested 2,097,152 bytes. After 20 rejected attempts, the helper stopped with:

```text
flame_salmon_1_split.z03 still has 1 missing or invalid parts
```

The range was divided into two independently verified 1 MiB requests and reconstructed:

```powershell
& curl.exe --http1.1 --fail --silent --show-error --range '301989888-303038463' --output '...\part-144-a.bin' $url
& curl.exe --http1.1 --fail --silent --show-error --range '303038464-304087039' --output '...\part-144-b.bin' $url
```

Both files were exactly 1,048,576 bytes; the reconstructed `part-144.bin` was exactly 2,097,152 bytes.

For `sear_steak`, two ranges similarly returned complementary lengths (1,621,270 + 475,882, and 31,793 + 2,065,359). A direct unsplit subrange attempt became slow and was stopped after verifying its exact curl PID and parent wrapper. Created `complete_sear_boundary_parts.ps1` with `apply_patch`; it splits at the observed server boundaries and uses 256 KiB subranges without crossing them:

```powershell
& 'C:\Users\CLEFFFF\Documents\Codex\2026-09-21\ym\work\complete_sear_boundary_parts.ps1'
```

Result: two subranges timed out in round 1, both succeeded in round 2, and exact 2,097,152-byte parts 408 and 421 were reconstructed. The main downloader then completed successfully.

Final official-asset results:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `coffee_martini.zip` | 1,186,324,684 | `CBC31291CE143E6F31A00F23A049663A61CC8793AF8DAB259DBC6E338A676816` |
| `cook_spinach.zip` | 1,212,423,873 | `19689AA099D673FCF4816DCAD2C1298DF609859E4B4D95A4983DBE48CCAB669D` |
| `flame_salmon_1_split.z01` | 1,572,864,000 | `C4A7FED9D7F0F17E800F419BA64EBEC31A23BDAE19A994EC9AEF75D2CC3A89A8` |
| `flame_salmon_1_split.z02` | 1,572,864,000 | `68B2701596DA02E34B5340FCE741FC7074A8F2A0287E450DA49A7C35441ADB44` |
| `flame_salmon_1_split.z03` | 1,572,864,000 | `EACD04F1C6AA7BCD9BF3FDFFE57FEDF305FCEFF830B236D12CEFAADA5CA7FF3F` |
| `flame_salmon_1_split.zip` | 272,910,788 | `00BEDB9478FDFAF8C0C1DC2BBF20A375EB47B509B976C47189C1431837696E57` |
| `flame_steak.zip` | 1,199,567,884 | `515C6BBB039250A4F7D379342ABB9DB81A12204B9FD987FA0353832A8C982DA4` |
| `sear_steak.zip` | 1,188,520,069 | `B53D9C444AB23D468F9D5DA0DCBC5AB68CB70702BB03F9664CD686E9EFEBC525` |

### 39. Archive testing and extraction

The manifest and archive directory were inspected. Windows tar successfully listed the four ordinary archives but rejected the split archive:

```powershell
Get-Content -LiteralPath 'E:\4DGS\downloads\n3v-all\download_manifest.json' -Raw
foreach($n in 'coffee_martini.zip','cook_spinach.zip','flame_steak.zip','sear_steak.zip','flame_salmon_1_split.zip'){ & tar.exe -tf "E:\4DGS\downloads\n3v-all\$n" | Select-Object -First 8 }
```

Error for the split archive: `tar.exe: Error opening archive: Unrecognized archive format`.

The ordinary archives were extracted concurrently:

```powershell
$archives=@('coffee_martini.zip','cook_spinach.zip','flame_steak.zip','sear_steak.zip')
$archives | ForEach-Object -Parallel { & tar.exe -xf "E:\4DGS\downloads\n3v-all\$_" -C 'E:\4DGS\data'; if($LASTEXITCODE -ne 0){throw "tar failed: $_ exit $LASTEXITCODE"} } -ThrottleLimit 4
```

Result: all four succeeded.

To obtain multi-volume support without installing software system-wide, the current Winget package record was inspected:

```powershell
winget show --id 7zip.7zip --exact --accept-source-agreements
```

It identified official 7-Zip 26.03 MSI URL `https://www.7-zip.org/a/7z2603-x64.msi` and SHA-256 `c0680064d698a62dd4a5a47f403db356a6531a5473e4c4b1d090ea2590513926`.

```powershell
curl.exe --http1.1 -L --fail --retry 10 --retry-all-errors --retry-delay 2 -o 'E:\4DGS\downloads\7z2603-x64.msi' 'https://www.7-zip.org/a/7z2603-x64.msi'
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\4DGS\downloads\7z2603-x64.msi'
$args=@('/a','E:\4DGS\downloads\7z2603-x64.msi','/qn','TARGETDIR=E:\4DGS\tools\7zip-26.03')
$p=Start-Process -FilePath 'msiexec.exe' -ArgumentList $args -Wait -PassThru -WindowStyle Hidden
```

Result: the 2,007,040-byte MSI hash matched exactly; administrative extraction returned exit code 0 and produced local `E:\4DGS\tools\7zip-26.03\Files\7-Zip\7z.exe` without a system-wide install.

```powershell
& 'E:\4DGS\tools\7zip-26.03\Files\7-Zip\7z.exe' t 'E:\4DGS\downloads\n3v-all\flame_salmon_1_split.zip'
& 'E:\4DGS\tools\7zip-26.03\Files\7-Zip\7z.exe' x 'E:\4DGS\downloads\n3v-all\flame_salmon_1_split.zip' '-oE:\4DGS\data' -y
```

Result: both commands returned exit code 0 and `Everything is Ok`. 7-Zip reported the expected four-volume Zip64 archive, total physical size 4,991,502,788 bytes, 20 entries, and uncompressed size 4,990,902,896 bytes. It also printed `Headers Error` as a warning for the split Zip64 headers; data testing and extraction nevertheless completed successfully.

### 40. Validate extracted N3V scene structure and media metadata

An initial PowerShell inventory expression had an empty-pipe parser error after a `foreach` block. It was corrected by assigning the loop output to `$rows` and piping `$rows` separately.

```powershell
$rows=foreach($scene in 'coffee_martini','cook_spinach','flame_salmon_1','flame_steak','sear_steak'){ $p="E:\4DGS\data\$scene"; $files=Get-ChildItem -LiteralPath $p -File; [pscustomobject]@{Scene=$scene;Files=$files.Count;Videos=($files|Where-Object Extension -eq '.mp4').Count;Bytes=($files|Measure-Object Length -Sum).Sum;HasPoses=(Test-Path "$p\poses_bounds.npy")} }
$rows | Format-Table -AutoSize
```

All scenes contain `poses_bounds.npy`. Media headers and poses were then checked with the configured Ex4DGS environment:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\CLEFFFF\anaconda3\envs\Ex4DGS\python.exe' -c "import cv2,numpy as np,pathlib,json; root=pathlib.Path(r'E:\4DGS\data'); scenes=['coffee_martini','cook_spinach','flame_salmon_1','flame_steak','sear_steak']; out={}; ...; print(json.dumps(out,indent=2))"
```

Result:

| Scene | Cameras / pose rows | Frames per video | Resolution | FPS |
|---|---:|---:|---:|---:|
| `coffee_martini` | 18 | 300 | 2704×2028 | 30 |
| `cook_spinach` | 21 | 300 | 2704×2028 | 30 |
| `flame_salmon_1` | 19 | 1200 | 2704×2028 | 30 |
| `flame_steak` | 21 | 300 | 2704×2028 | 30 |
| `sear_steak` | 21 | 300 | 2704×2028 | 30 |

Every MP4 opened successfully. For every scene, `poses_bounds.npy` has shape `(camera_count, 17)`, exactly matching the number of videos.

### 41. Final archive CRC, extracted-size, source-state, and disk checks

```powershell
$seven='E:\4DGS\tools\7zip-26.03\Files\7-Zip\7z.exe'
$archives='coffee_martini.zip','cook_spinach.zip','flame_steak.zip','sear_steak.zip'
$archives | ForEach-Object -Parallel { & $using:seven t "E:\4DGS\downloads\n3v-all\$_" } -ThrottleLimit 4
Get-ChildItem -LiteralPath 'E:\4DGS\data' -Directory | Where-Object Name -in 'coffee_martini','cook_spinach','cut_roasted_beef','flame_salmon_1','flame_steak','sear_steak' | ForEach-Object { [pscustomobject]@{Scene=$_.Name;Bytes=(Get-ChildItem -LiteralPath $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum;Files=(Get-ChildItem -LiteralPath $_.FullName -Recurse -File).Count} }
Get-PSDrive -Name E | Format-List Name,Used,Free
git -C 'E:\4DGS' diff --exit-code
git -C 'E:\4DGS' ls-files -m
```

Result: every ordinary ZIP returned exit code 0 and `Everything is Ok`; the split ZIP had already passed the same full 7-Zip test. Extracted raw sizes are 1,186,189,096; 1,212,300,319; 4,990,902,896; 1,199,446,151; and 1,188,401,508 bytes respectively. E: retained 1,383,864,942,592 bytes free. `git diff --exit-code` succeeded and `git ls-files -m` returned nothing: no tracked algorithm source was modified.
