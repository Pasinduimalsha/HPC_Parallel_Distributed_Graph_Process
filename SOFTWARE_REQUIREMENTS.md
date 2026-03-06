# Software Requirements

**EE7218/EC7207 HPC Project – Group 24**

---

## macOS – Software to Install

| Software | Purpose | Install Command |
|----------|---------|-----------------|
| **Xcode Command Line Tools** | C compiler (Clang) | `xcode-select --install` |
| **Homebrew** | Package manager | [brew.sh](https://brew.sh) |
| **libomp** | OpenMP support (Apple Clang lacks it) | `brew install libomp` |
| **Open-MPI** | MPI for distributed runs | `brew install open-mpi` |
| **CMake** (optional) | Cross-platform build | `brew install cmake` |

### Notes for macOS
- **CUDA / Hybrid:** Not supported on macOS. Apple removed NVIDIA GPU support. Use Serial, OpenMP, and MPI only.
- **OpenMP:** Apple Clang does not include OpenMP. You must install `libomp` via Homebrew.

---

## Windows – Software to Install

| Software | Purpose | Download / Install |
|----------|---------|---------------------|
| **Visual Studio 2022** (or 2019) | C compiler, OpenMP | [visualstudio.com](https://visualstudio.microsoft.com/) – install "Desktop development with C++" workload |
| **Microsoft MPI (MS-MPI)** | MPI for distributed runs | [Microsoft MPI Downloads](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) |
| **CMake** | Cross-platform build | [cmake.org](https://cmake.org/download/) or `winget install Kitware.CMake` |
| **CUDA Toolkit** (optional) | Hybrid (GPU) builds | [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) – only if you have an NVIDIA GPU |

### Notes for Windows
- **OpenMP:** Included with Visual Studio when you install the C++ workload.
- **MPI:** Use MS-MPI (Microsoft MPI) for `mpiexec` and MPI libraries.
- **Hybrid:** Requires an NVIDIA GPU and CUDA Toolkit.

---

## Summary by Implementation

| Implementation | macOS | Windows |
|----------------|-------|---------|
| Serial | ✅ Xcode CLT | ✅ Visual Studio |
| OpenMP | ✅ libomp | ✅ Visual Studio (built-in) |
| MPI | ✅ open-mpi | ✅ MS-MPI |
| Hybrid (CUDA) | ❌ Not supported | ✅ CUDA Toolkit + NVIDIA GPU |

---

## Quick Install Commands

### macOS (using Homebrew)
```bash
# Install all required (except CUDA – not available on Mac)
brew install libomp open-mpi cmake
```

### Windows (using winget)
```powershell
winget install Microsoft.VisualStudio.2022.Community
winget install Kitware.CMake
# MS-MPI: download and install from Microsoft's site
```
