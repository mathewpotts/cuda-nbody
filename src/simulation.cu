#include "simulation.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

void checkCuda(cudaError_t err, const char* message)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void initializeBodies(std::vector<Body>& bodies)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 1.5f);

    for (size_t i = 0; i < bodies.size(); ++i)
    {
        bodies[i].pos = make_float3(
            posDist(rng),
            posDist(rng),
            posDist(rng));

        bodies[i].vel = make_float3(0.0f, 0.0f, 0.0f);
        bodies[i].mass = massDist(rng);
    }
}
