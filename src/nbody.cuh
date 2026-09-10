#pragma once

#include <cuda_runtime.h>

inline constexpr float G = 6.67430e-11f;
inline constexpr float INITIAL_RADIUS = 1.0e8f;
inline constexpr float BASE_MASS = 1.0e20f;
inline constexpr float SOFTENING = 1.0e5f;
inline constexpr float EPS2 = SOFTENING * SOFTENING;

struct Body {
    float3 pos;
    float3 vel;
    float mass;
};

__global__ void computeAccelerations(const Body* bodies, float3* accel, int N);
__global__ void kickDrift(Body* bodies, const float3* accel, float dt, int N);
__global__ void finalKick(Body* bodies, const float3* accel, float dt, int N);
__global__ void writeRenderPositions(const Body* bodies, float* positions,
                                     float renderScale, int N);
__global__ void accumulateDiagnostics(const Body* bodies, double* accumulator, int N);
