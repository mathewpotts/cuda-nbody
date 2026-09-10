#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "nbody.cuh"

struct Diagnostics {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;

    double momentumX = 0.0;
    double momentumY = 0.0;
    double momentumZ = 0.0;

    double centerOfMassX = 0.0;
    double centerOfMassY = 0.0;
    double centerOfMassZ = 0.0;

    double totalMass = 0.0;
};

void checkCuda(cudaError_t result, const char* message);
void initializeBodies(std::vector<Body>& bodies);

Diagnostics calculateDiagnostics(
    const Body* d_bodies,
    double* d_accumulator,
    int blocks,
    int threads,
    int N
);
