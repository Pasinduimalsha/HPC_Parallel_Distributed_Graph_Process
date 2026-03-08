@echo off
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" (
    call "%VCVARS%"
) else (
    echo vcvars64.bat not found at %VCVARS%
)
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files\Microsoft MPI\Bin"
cd /d "%~dp0"
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% neq 0 (
    cmake .. -G "Visual Studio 16 2019" -A x64
)
cmake --build . --config Release
cd ..
echo Build complete.
