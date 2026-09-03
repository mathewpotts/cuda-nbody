#include <cuda_runtime.h>
#include <cmath>

constexpr float G = 1.0f;
constexpr float EPS2 = 1e-4f;


struct Body
{
    float3 pos;
    float3 vel;
    float mass;
};


// ---------------------------------------------------------
// Calculate gravitational acceleration
// ---------------------------------------------------------

__global__
void computeAccelerations(
    const Body* bodies,
    float3* accel,
    int N)
{
    extern __shared__ Body sharedBodies[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    float3 pi = bodies[i].pos;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile = 0; tile < N; tile += blockDim.x)
    {
        int j = tile + threadIdx.x;

        if (j < N)
        {
            sharedBodies[threadIdx.x] = bodies[j];
        }

        __syncthreads();

        int tileSize = min(blockDim.x, N - tile);

        for (int k = 0; k < tileSize; ++k)
        {
            int globalJ = tile + k;

            if (globalJ == i)
                continue;

            float dx = sharedBodies[k].pos.x - pi.x;
            float dy = sharedBodies[k].pos.y - pi.y;
            float dz = sharedBodies[k].pos.z - pi.z;

            float r2 = dx * dx + dy * dy + dz * dz + EPS2;
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;

            float s = G * sharedBodies[k].mass * invR3;

            ax += dx * s;
            ay += dy * s;
            az += dz * s;
        }

        __syncthreads();
    }

    accel[i] = make_float3(ax, ay, az);
}


// ---------------------------------------------------------
// Update velocity and position
// ---------------------------------------------------------

__global__
void integrate(
    Body* bodies,
    const float3* accel,
    float dt,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    Body b = bodies[i];

    // Update velocity
    b.vel.x += accel[i].x * dt;
    b.vel.y += accel[i].y * dt;
    b.vel.z += accel[i].z * dt;

    // Update position
    b.pos.x += b.vel.x * dt;
    b.pos.y += b.vel.y * dt;
    b.pos.z += b.vel.z * dt;

    bodies[i] = b;
}