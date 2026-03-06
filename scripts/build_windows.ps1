# PowerShell build script for Windows
# Requires: Visual Studio, CMake, (optional) MS-MPI, (optional) CUDA

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$BuildDir = "build"
if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir }
Set-Location $BuildDir

# Try Ninja or default generator
$generator = "Visual Studio 17 2022"
$arch = "x64"
cmake .. -G $generator -A $arch
if ($LASTEXITCODE -ne 0) {
    $generator = "Visual Studio 16 2019"
    cmake .. -G $generator -A $arch
}

cmake --build . --config Release
Set-Location ..
Write-Host "Build complete. Executables in build\Release\"
