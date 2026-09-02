#pragma once

#include <cuda_runtime.h>

constexpr float G = 1.0f;
constexpr float EPS2 = 1e-4f;

struct Body
{
    float3 pos;
    float3 vel;
    float mass;
};

__global__ void computeAccelerations(
    const Body* bodies,
    float3* accel,
    int N);

__global__ void integrate(
    Body* bodies,
    const float3* accel,
    float dt,
    int N);