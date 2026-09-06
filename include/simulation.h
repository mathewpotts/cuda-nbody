#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "nbody.cuh"

void checkCuda(cudaError_t err, const char* message);
void initializeBodies(std::vector<Body>& bodies);
