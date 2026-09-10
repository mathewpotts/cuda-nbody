#include "simulation.h"

#include <cstdio>
#include <cstdlib>
#include <random>

void checkCuda(cudaError_t result, const char* message) {
    if (result == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "CUDA error: %s: %s\n",
                 message, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
}

void initializeBodies(std::vector<Body>& bodies) {
    std::mt19937 rng(42);

    std::uniform_real_distribution<float> positionDist(
        -INITIAL_RADIUS, INITIAL_RADIUS
    );

    std::uniform_real_distribution<float> massDist(
        0.5f * BASE_MASS, 1.5f * BASE_MASS
    );

    for (Body& body : bodies) {
        float x;
        float y;
        float z;

        do {
            x = positionDist(rng);
            y = positionDist(rng);
            z = positionDist(rng);
        } while (x * x + y * y + z * z >
                 INITIAL_RADIUS * INITIAL_RADIUS);

        body.pos = make_float3(x, y, z);
        body.vel = make_float3(0.0f, 0.0f, 0.0f);
        body.mass = massDist(rng);
    }
}

Diagnostics calculateDiagnostics(
    const Body* d_bodies,
    double* d_accumulator,
    int blocks,
    int threads,
    int N
) {
    constexpr int NUM_VALUES = 9;

    checkCuda(
        cudaMemset(d_accumulator, 0, NUM_VALUES * sizeof(double)),
        "cudaMemset diagnostics"
    );

    accumulateDiagnostics<<<blocks, threads>>>(d_bodies, d_accumulator, N);
    checkCuda(cudaGetLastError(), "accumulateDiagnostics launch");

    double values[NUM_VALUES]{};

    checkCuda(
        cudaMemcpy(values, d_accumulator, NUM_VALUES * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "diagnostics device-to-host copy"
    );

    Diagnostics d;

    d.kineticEnergy = values[0];
    d.potentialEnergy = values[1];
    d.totalEnergy = values[0] + values[1];

    d.momentumX = values[2];
    d.momentumY = values[3];
    d.momentumZ = values[4];

    d.totalMass = values[8];

    if (d.totalMass != 0.0) {
        d.centerOfMassX = values[5] / d.totalMass;
        d.centerOfMassY = values[6] / d.totalMass;
        d.centerOfMassZ = values[7] / d.totalMass;
    }

    return d;
}
