# CUDA N-Body

Complete Windows/CUDA/OpenGL project.

Features: real SI `G`, leapfrog integration, shared `float4` force tiles, GPU diagnostics, CUDA/OpenGL VBO interop, rewind history, camera controls.

## Dependencies

Use your existing vcpkg tree at:

`C:\Users\Maddie\Documents\GitHub\vcpkg`

Install once:

```powershell
& "C:\Users\Maddie\Documents\GitHub\vcpkg\vcpkg.exe" install glfw3:x64-windows glad:x64-windows
```

## Configure from `cuda-nbody\build`

```powershell
Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue; Remove-Item CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue; cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/Users/Maddie/Documents/GitHub/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

## Build

```powershell
cmake --build . --config Release
```

## Run

```powershell
.\Release\nbody.exe 256
```
