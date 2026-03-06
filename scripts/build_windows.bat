@echo off
REM Build script for Windows - uses CMake
REM Requires: Visual Studio, CMake, (optional) MS-MPI, (optional) CUDA

cd /d "%~dp0\.."

if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% neq 0 (
    echo Try: cmake .. -G "Visual Studio 16 2019" -A x64
    cmake .. -G "Visual Studio 16 2019" -A x64
)

cmake --build . --config Release
cd ..
echo Build complete. Executables in build\Release\
