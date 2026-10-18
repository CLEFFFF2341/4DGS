$ErrorActionPreference = 'Stop'
function Invoke-TaskNative {
    param([string]$Executable, [string[]]$Arguments, [string]$Log)
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments *> $Log
        $script:TaskNativeExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($script:TaskNativeExit -ne 0) { throw "Command failed ($script:TaskNativeExit): $Log" }
}
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.conda\nerficg\python.exe'
$vsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path -LiteralPath $vsWhere)) { throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++ first.' }
$vsRoot = & $vsWhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsRoot) { throw 'MSVC v143 compiler installation not found.' }
$devCmd = Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
$taskDevCommand = 'call "' + $devCmd + '" -arch=x64 -host_arch=x64 >nul && set'
cmd.exe /d /c $taskDevCommand | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'MSVC environment activation failed: cl.exe is not on PATH.' }
$env:DISTUTILS_USE_SDK = '1'
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:MAX_JOBS = '4'
$env:TORCH_HOME = Join-Path $taskRoot '.cache\torch'
if (!$env:CUDA_PATH) { $env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' }
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$taskRoot\.conda\nerficg;$taskRoot\.conda\nerficg\Library\bin;$taskRoot\.conda\nerficg\Scripts;$env:PATH"
$env:INCLUDE = "$taskRoot\.conda\nerficg\Library\include;$taskRoot\.conda\nerficg\Library\include\torch\csrc\api\include;$env:INCLUDE"
$env:LIB = "$taskRoot\.conda\nerficg\Library\lib;$env:LIB"
if (!(Test-Path -LiteralPath "$env:CUDA_PATH\bin\nvcc.exe")) { throw 'CUDA Toolkit 12.8 nvcc was not found.' }
& $taskPython -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw 'PyTorch CUDA preflight failed. Install deployment/requirements-official.txt first.' }
Invoke-TaskNative $taskPython @('scripts/install.py', '-m', 'FasterGS4D') 'deployment/extensions.log'
# install.py may report an error while returning zero, so import the method explicitly.
& $taskPython -c "import sys; sys.path.insert(0,'src'); import Methods.FasterGS4D"
if ($LASTEXITCODE) { throw 'FasterGS4D import failed.' }
if (!(Test-Path -LiteralPath 'deployment/dataset_settings.json')) { throw 'Complete prepare_coffee.py and dataset verification first.' }
if (!(Test-Path -LiteralPath 'deployment/image_cache.json')) { throw 'Run deployment/cache_images.py and finish all 5400 equality checks first.' }
Invoke-TaskNative $taskPython @('-u', 'deployment/run_with_cache.py', 'scripts/train.py', '-c', 'configs/fastergs4d_coffee_martini.yaml') 'deployment/train.log'
$run = Get-ChildItem output/FasterGS4D -Directory | Where-Object { $_.Name -like '*fastergs4d_coffee_martini*' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'checkpoints\final.pt')) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$run) { throw 'No completed coffee_martini checkpoint found.' }
Invoke-TaskNative $taskPython @('-u', 'deployment/run_with_cache.py', 'scripts/inference.py', '-d', $run.FullName, '-s', 'test', '-m') 'deployment/evaluate.log'
$ffmpeg = Get-ChildItem tools -Filter ffmpeg.exe -Recurse | Select-Object -First 1
$rgb = Join-Path $run.FullName 'inference\test_30000\rgb\%05d.png'
$video = Join-Path $run.FullName 'coffee_martini_cam00_dynamic.mp4'
Invoke-TaskNative $ffmpeg.FullName @('-y', '-framerate', '30', '-i', $rgb, '-frames:v', '300', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', $video) 'deployment/video.log'
Write-Output "Completed run: $($run.FullName)"
Write-Output "Dynamic held-out camera video: $video"
