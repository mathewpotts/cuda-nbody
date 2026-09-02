#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// For this initial version, use kernel implementation.
//
// Later we can replace this with:
// #include "nbody.cuh"
#include "nbody.cu"


// ---------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------

void checkCuda(cudaError_t err, const char* message)
{
    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA Error: %s: %s\n",
            message,
            cudaGetErrorString(err));

        exit(EXIT_FAILURE);
    }
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main()
{
    // Number of bodies
    const int N = 8192;

    // Number of simulation steps
    const int steps = 1000;

    // Time step
    const float dt = 0.001f;


    // -----------------------------------------------------
    // Create bodies on CPU
    // -----------------------------------------------------

    std::vector<Body> h_bodies(N);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float> posDist(
        -1.0f,
        1.0f);

    std::uniform_real_distribution<float> massDist(
        0.5f,
        1.5f);


    for (int i = 0; i < N; ++i)
    {
        h_bodies[i].pos = make_float3(
            posDist(rng),
            posDist(rng),
            posDist(rng));

        h_bodies[i].vel = make_float3(
            0.0f,
            0.0f,
            0.0f);

        h_bodies[i].mass = massDist(rng);
    }


    // -----------------------------------------------------
    // Allocate GPU memory
    // -----------------------------------------------------

    Body* d_bodies = nullptr;
    float3* d_accel = nullptr;

    checkCuda(
        cudaMalloc(
            &d_bodies,
            N * sizeof(Body)),
        "cudaMalloc d_bodies");

    checkCuda(
        cudaMalloc(
            &d_accel,
            N * sizeof(float3)),
        "cudaMalloc d_accel");


    // -----------------------------------------------------
    // Copy initial bodies CPU -> GPU
    // -----------------------------------------------------

    checkCuda(
        cudaMemcpy(
            d_bodies,
            h_bodies.data(),
            N * sizeof(Body),
            cudaMemcpyHostToDevice),
        "Copy bodies to GPU");


    // -----------------------------------------------------
    // CUDA launch configuration
    // -----------------------------------------------------

    const int threads = 256;

    const int blocks =
        (N + threads - 1) / threads;


    printf("Bodies:  %d\n", N);
    printf("Threads: %d\n", threads);
    printf("Blocks:  %d\n", blocks);
    printf("Steps:   %d\n\n", steps);


    // -----------------------------------------------------
    // Simulation
    // -----------------------------------------------------

    for (int step = 0; step < steps; ++step)
    {
        computeAccelerations<<<blocks, threads>>>(
            d_bodies,
            d_accel,
            N);

        checkCuda(
            cudaGetLastError(),
            "computeAccelerations");


        integrate<<<blocks, threads>>>(
            d_bodies,
            d_accel,
            dt,
            N);

        checkCuda(
            cudaGetLastError(),
            "integrate");


        if (step % 100 == 0)
        {
            checkCuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize");

            printf(
                "Step %d / %d\n",
                step,
                steps);
        }
    }


    // -----------------------------------------------------
    // Copy final state GPU -> CPU
    // -----------------------------------------------------

    checkCuda(
        cudaMemcpy(
            h_bodies.data(),
            d_bodies,
            N * sizeof(Body),
            cudaMemcpyDeviceToHost),
        "Copy bodies from GPU");


    // -----------------------------------------------------
    // Print some results
    // -----------------------------------------------------

    printf("\nFinal positions:\n\n");

    for (int i = 0; i < 10; ++i)
    {
        printf(
            "Body %4d : "
            "%10.6f "
            "%10.6f "
            "%10.6f\n",
            i,
            h_bodies[i].pos.x,
            h_bodies[i].pos.y,
            h_bodies[i].pos.z);
    }


    // -----------------------------------------------------
    // Cleanup
    // -----------------------------------------------------

    cudaFree(d_accel);
    cudaFree(d_bodies);

    return 0;
}