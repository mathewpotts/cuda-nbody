#include "nbody.cuh"

#include <cmath>

__global__ void computeAccelerations(const Body* bodies, float3* accel, int N) {
    extern __shared__ float4 sharedBodies[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < N;

    float3 pi = make_float3(0.0f, 0.0f, 0.0f);
    if (active) {
        pi = bodies[i].pos;
    }

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile = 0; tile < N; tile += blockDim.x) {
        const int j = tile + threadIdx.x;

        if (j < N) {
            const Body b = bodies[j];
            sharedBodies[threadIdx.x] =
                make_float4(b.pos.x, b.pos.y, b.pos.z, b.mass);
        }

        __syncthreads();

        if (active) {
            const int tileSize = min(blockDim.x, N - tile);

            for (int k = 0; k < tileSize; ++k) {
                const int globalJ = tile + k;
                if (globalJ == i) {
                    continue;
                }

                const float4 bj = sharedBodies[k];

                const float dx = bj.x - pi.x;
                const float dy = bj.y - pi.y;
                const float dz = bj.z - pi.z;

                const float r2 = dx * dx + dy * dy + dz * dz + EPS2;
                const float invR = rsqrtf(r2);
                const float invR3 = invR * invR * invR;
                const float scale = G * bj.w * invR3;

                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }

        __syncthreads();
    }

    if (active) {
        accel[i] = make_float3(ax, ay, az);
    }
}

__global__ void kickDrift(Body* bodies, const float3* accel, float dt, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    Body b = bodies[i];
    const float halfDt = 0.5f * dt;

    b.vel.x += accel[i].x * halfDt;
    b.vel.y += accel[i].y * halfDt;
    b.vel.z += accel[i].z * halfDt;

    b.pos.x += b.vel.x * dt;
    b.pos.y += b.vel.y * dt;
    b.pos.z += b.vel.z * dt;

    bodies[i] = b;
}

__global__ void finalKick(Body* bodies, const float3* accel, float dt, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    Body b = bodies[i];
    const float halfDt = 0.5f * dt;

    b.vel.x += accel[i].x * halfDt;
    b.vel.y += accel[i].y * halfDt;
    b.vel.z += accel[i].z * halfDt;

    bodies[i] = b;
}

__global__ void writeRenderPositions(const Body* bodies, float* positions,
                                     float renderScale, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    positions[3 * i + 0] = bodies[i].pos.x * renderScale;
    positions[3 * i + 1] = bodies[i].pos.y * renderScale;
    positions[3 * i + 2] = bodies[i].pos.z * renderScale;
}

__global__ void accumulateDiagnostics(const Body* bodies, double* accumulator, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const Body bi = bodies[i];

    const double mass = static_cast<double>(bi.mass);
    const double vx = static_cast<double>(bi.vel.x);
    const double vy = static_cast<double>(bi.vel.y);
    const double vz = static_cast<double>(bi.vel.z);

    const double kinetic = 0.5 * mass * (vx * vx + vy * vy + vz * vz);

    double potential = 0.0;

    for (int j = i + 1; j < N; ++j) {
        const Body bj = bodies[j];

        const double dx =
            static_cast<double>(bj.pos.x) - static_cast<double>(bi.pos.x);
        const double dy =
            static_cast<double>(bj.pos.y) - static_cast<double>(bi.pos.y);
        const double dz =
            static_cast<double>(bj.pos.z) - static_cast<double>(bi.pos.z);

        const double r = sqrt(
            dx * dx + dy * dy + dz * dz + static_cast<double>(EPS2)
        );

        potential -= static_cast<double>(G) * mass *
                     static_cast<double>(bj.mass) / r;
    }

    atomicAdd(&accumulator[0], kinetic);
    atomicAdd(&accumulator[1], potential);

    atomicAdd(&accumulator[2], mass * vx);
    atomicAdd(&accumulator[3], mass * vy);
    atomicAdd(&accumulator[4], mass * vz);

    atomicAdd(&accumulator[5], mass * static_cast<double>(bi.pos.x));
    atomicAdd(&accumulator[6], mass * static_cast<double>(bi.pos.y));
    atomicAdd(&accumulator[7], mass * static_cast<double>(bi.pos.z));

    atomicAdd(&accumulator[8], mass);
}
