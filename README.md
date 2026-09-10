# CUDA N-Body

Complete Windows/CUDA/OpenGL project.

Features: real SI `G`, leapfrog integration, shared `float4` force tiles, GPU diagnostics, CUDA/OpenGL VBO interop, rewind history, camera controls.

## Current Simulation

* Bodies: **16,384**
* Steps: **2,000**
* Default threads per block: **256**
* Timestep: **100 seconds**
* Gravitational constant: **6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²**
* Initial radius: **100,000 km**
* Initial body mass: **0.5 × 10²⁰ to 1.5 × 10²⁰ kg**
* Initial velocity: **0 m/s**
* Gravitational softening: **100 km**

## Dependencies

Use your existing vcpkg tree at:

`C:\Users\Maddie\Documents\GitHub\vcpkg\`

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

The optional argument specifies the number of CUDA threads per block.

For example:

```powershell
.\Release\nbody.exe 128
.\Release\nbody.exe 256
.\Release\nbody.exe 512
```

## Controls

* **Space** - Play/pause simulation
* **Right Arrow** - Advance one step
* **Left Arrow** - Rewind one step
* **Left Mouse + Drag** - Rotate camera
* **Mouse Wheel** - Zoom
* **ESC** - Exit

## Physics

The simulation uses SI units throughout:

* Position: meters
* Velocity: meters per second
* Mass: kilograms
* Time: seconds
* Energy: joules

The gravitational force calculation uses shared-memory `float4` tiles. Each tile stores the source body's `x`, `y`, `z`, and mass, avoiding unnecessary velocity loads during the force calculation.

The simulation uses leapfrog integration:

1. Half velocity kick
2. Full position drift
3. Recalculate gravitational acceleration
4. Final half velocity kick

This provides better long-term energy behavior than the previous Euler integration.

## Diagnostics

The GPU calculates:

* Kinetic energy
* Gravitational potential energy
* Total energy
* Energy drift
* Total momentum
* Center of mass
* Center-of-mass drift

Diagnostics are printed periodically while the simulation runs.

## CUDA/OpenGL Interop

Newly calculated positions are written directly from CUDA into the OpenGL vertex buffer using CUDA/OpenGL interoperability:

```text
CUDA Body Data
      |
      v
CUDA Kernel
      |
      v
OpenGL VBO
      |
      v
glDrawArrays
```

This avoids copying newly calculated rendering positions from the GPU to the CPU and then back to the GPU.

CPU copies are still retained for simulation rewind history.

## Performance

The gravitational force calculation is an O(N²) direct N-body calculation.

With **16,384 bodies**, each timestep evaluates approximately:

```text
16,384² = 268,435,456
```

body interactions.

Across **2,000 steps**, the full simulation performs approximately:

```text
536,870,912,000
```

body interactions.

CUDA event timing is used to measure the physics kernels independently from OpenGL rendering and VSync.
