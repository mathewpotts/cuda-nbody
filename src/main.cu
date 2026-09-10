#define GLFW_INCLUDE_NONE

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nbody.cuh"
#include "shader.h"
#include "simulation.h"

namespace {

constexpr int NUM_BODIES = 16384;
constexpr int NUM_STEPS = 2000;
constexpr float DT = 100.0f;
constexpr int DIAGNOSTIC_INTERVAL = 10;

float gYaw = 0.0f;
float gPitch = 0.0f;
float gZoom = 1.0f;

bool gDragging = false;
double gLastMouseX = 0.0;
double gLastMouseY = 0.0;

bool gPlaying = false;
bool gPreviousSpace = false;
bool gPreviousRight = false;
bool gPreviousLeft = false;

void scrollCallback(GLFWwindow*, double, double yOffset) {
    constexpr float ZOOM_FACTOR = 1.15f;

    if (yOffset > 0.0) {
        gZoom *= ZOOM_FACTOR;
    } else if (yOffset < 0.0) {
        gZoom /= ZOOM_FACTOR;
    }

    gZoom = std::clamp(gZoom, 0.05f, 100.0f);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }

    if (action == GLFW_PRESS) {
        gDragging = true;
        glfwGetCursorPos(window, &gLastMouseX, &gLastMouseY);
    } else if (action == GLFW_RELEASE) {
        gDragging = false;
    }
}

void cursorPositionCallback(GLFWwindow*, double xpos, double ypos) {
    if (!gDragging) {
        return;
    }

    const double dx = xpos - gLastMouseX;
    const double dy = ypos - gLastMouseY;

    gLastMouseX = xpos;
    gLastMouseY = ypos;

    constexpr float SENSITIVITY = 0.005f;

    gYaw += static_cast<float>(dx) * SENSITIVITY;
    gPitch += static_cast<float>(dy) * SENSITIVITY;
    gPitch = std::clamp(gPitch, -1.55f, 1.55f);
}

void fillRenderPositions(
    const std::vector<Body>& bodies,
    std::vector<float>& positions,
    float renderScale
) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        positions[3 * i + 0] = bodies[i].pos.x * renderScale;
        positions[3 * i + 1] = bodies[i].pos.y * renderScale;
        positions[3 * i + 2] = bodies[i].pos.z * renderScale;
    }
}

void uploadHistoryFrame(
    unsigned int vbo,
    const std::vector<Body>& bodies,
    std::vector<float>& positions,
    float renderScale
) {
    fillRenderPositions(bodies, positions, renderScale);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        positions.size() * sizeof(float),
        positions.data()
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void updateVboFromCuda(
    cudaGraphicsResource* cudaVbo,
    const Body* dBodies,
    int blocks,
    int threads,
    int N,
    float renderScale
) {
    checkCuda(
        cudaGraphicsMapResources(1, &cudaVbo, 0),
        "cudaGraphicsMapResources"
    );

    float* dPositions = nullptr;
    size_t mappedBytes = 0;

    checkCuda(
        cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&dPositions),
            &mappedBytes,
            cudaVbo
        ),
        "cudaGraphicsResourceGetMappedPointer"
    );

    const size_t requiredBytes =
        static_cast<size_t>(N) * 3 * sizeof(float);

    if (mappedBytes < requiredBytes) {
        std::fprintf(stderr, "Mapped VBO is smaller than expected.\n");
        std::exit(EXIT_FAILURE);
    }

    writeRenderPositions<<<blocks, threads>>>(
        dBodies,
        dPositions,
        renderScale,
        N
    );

    checkCuda(cudaGetLastError(), "writeRenderPositions launch");

    checkCuda(
        cudaGraphicsUnmapResources(1, &cudaVbo, 0),
        "cudaGraphicsUnmapResources"
    );
}

void printDiagnostics(
    const Diagnostics& current,
    const Diagnostics& initial,
    int step
) {
    const double simulatedHours =
        static_cast<double>(step) * DT / 3600.0;

    const double momentumMagnitude = std::sqrt(
        current.momentumX * current.momentumX +
        current.momentumY * current.momentumY +
        current.momentumZ * current.momentumZ
    );

    const double dx = current.centerOfMassX - initial.centerOfMassX;
    const double dy = current.centerOfMassY - initial.centerOfMassY;
    const double dz = current.centerOfMassZ - initial.centerOfMassZ;
    const double comDrift = std::sqrt(dx * dx + dy * dy + dz * dz);

    double energyDriftPercent = 0.0;

    if (std::abs(initial.totalEnergy) > 0.0) {
        energyDriftPercent =
            100.0 * (current.totalEnergy - initial.totalEnergy) /
            std::abs(initial.totalEnergy);
    }

    std::printf(
        "\n\nPhysics diagnostics\n"
        "Step:              %d\n"
        "Simulation time:   %.3f hours\n"
        "Kinetic energy:    %.8e J\n"
        "Potential energy:  %.8e J\n"
        "Total energy:      %.8e J\n"
        "Energy drift:      %.6f %%\n"
        "|Momentum|:        %.8e kg m/s\n"
        "Center of mass:    %.6e %.6e %.6e m\n"
        "COM drift:         %.6e m\n\n",
        step,
        simulatedHours,
        current.kineticEnergy,
        current.potentialEnergy,
        current.totalEnergy,
        energyDriftPercent,
        momentumMagnitude,
        current.centerOfMassX,
        current.centerOfMassY,
        current.centerOfMassZ,
        comDrift
    );
}

}  // namespace

int main(int argc, char* argv[]) {
    int threads = 256;

    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [threads_per_block]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        threads = std::atoi(argv[1]);
    }

    if (threads <= 0 || threads > 1024) {
        std::fprintf(stderr, "threads_per_block must be 1..1024\n");
        return EXIT_FAILURE;
    }

    const int blocks = (NUM_BODIES + threads - 1) / threads;
    const size_t sharedBytes =
        static_cast<size_t>(threads) * sizeof(float4);

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        1000, 800, "CUDA N-Body", nullptr, nullptr
    );

    if (!window) {
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    if (!gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "Failed to initialize GLAD.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    std::printf("OpenGL version: %s\n", glGetString(GL_VERSION));

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_PROGRAM_POINT_SIZE);

    ShaderProgram shaderProgram = createShaderProgram();

    std::vector<Body> hBodies(NUM_BODIES);
    initializeBodies(hBodies);

    std::vector<std::vector<Body>> history;
    history.reserve(NUM_STEPS + 1);
    history.push_back(hBodies);

    int displayStep = 0;
    int newestStep = 0;

    Body* dBodies = nullptr;
    float3* dAccel = nullptr;
    double* dDiagnostics = nullptr;

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dBodies),
            NUM_BODIES * sizeof(Body)
        ),
        "cudaMalloc dBodies"
    );

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dAccel),
            NUM_BODIES * sizeof(float3)
        ),
        "cudaMalloc dAccel"
    );

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dDiagnostics),
            9 * sizeof(double)
        ),
        "cudaMalloc dDiagnostics"
    );

    checkCuda(
        cudaMemcpy(
            dBodies,
            hBodies.data(),
            NUM_BODIES * sizeof(Body),
            cudaMemcpyHostToDevice
        ),
        "initial body upload"
    );

    computeAccelerations<<<blocks, threads, sharedBytes>>>(
        dBodies,
        dAccel,
        NUM_BODIES
    );

    checkCuda(cudaGetLastError(), "initial acceleration launch");
    checkCuda(cudaDeviceSynchronize(), "initial acceleration sync");

    const Diagnostics initialDiagnostics = calculateDiagnostics(
        dBodies,
        dDiagnostics,
        blocks,
        threads,
        NUM_BODIES
    );

    printDiagnostics(initialDiagnostics, initialDiagnostics, 0);

    const float renderScale = 0.75f / INITIAL_RADIUS;
    std::vector<float> positions(NUM_BODIES * 3);

    fillRenderPositions(hBodies, positions, renderScale);

    unsigned int vao = 0;
    unsigned int vbo = 0;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        positions.size() * sizeof(float),
        positions.data(),
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        nullptr
    );

    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    cudaGraphicsResource* cudaVbo = nullptr;

    checkCuda(
        cudaGraphicsGLRegisterBuffer(
            &cudaVbo,
            vbo,
            cudaGraphicsMapFlagsWriteDiscard
        ),
        "cudaGraphicsGLRegisterBuffer"
    );

    cudaEvent_t kernelStart = nullptr;
    cudaEvent_t kernelStop = nullptr;

    checkCuda(cudaEventCreate(&kernelStart), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&kernelStop), "cudaEventCreate stop");

    float totalKernelMs = 0.0f;

    std::printf(
        "\nBodies:              %d\n"
        "Steps:               %d\n"
        "Threads/block:       %d\n"
        "Blocks:              %d\n"
        "Shared memory/block: %zu bytes\n"
        "dt:                  %.2f seconds\n\n",
        NUM_BODIES,
        NUM_STEPS,
        threads,
        blocks,
        sharedBytes,
        DT
    );

    std::printf(
        "Controls:\n"
        "  Space           Play/pause\n"
        "  Right Arrow     One step forward\n"
        "  Left Arrow      One step backward\n"
        "  Left drag       Rotate\n"
        "  Mouse wheel     Zoom\n"
        "  ESC             Exit\n\n"
    );

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        const bool spacePressed =
            glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        const bool rightPressed =
            glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
        const bool leftPressed =
            glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS;

        if (spacePressed && !gPreviousSpace) {
            gPlaying = !gPlaying;
        }

        bool stateChanged = false;

        if (leftPressed && !gPreviousLeft) {
            gPlaying = false;

            if (displayStep > 0) {
                --displayStep;
                hBodies = history[displayStep];

                uploadHistoryFrame(
                    vbo,
                    hBodies,
                    positions,
                    renderScale
                );

                stateChanged = true;
            }
        }

        bool advance = gPlaying;

        if (rightPressed && !gPreviousRight) {
            gPlaying = false;
            advance = true;
        }

        if (advance && displayStep < NUM_STEPS) {
            if (displayStep < newestStep) {
                ++displayStep;
                hBodies = history[displayStep];

                uploadHistoryFrame(
                    vbo,
                    hBodies,
                    positions,
                    renderScale
                );

                stateChanged = true;
            } else {
                checkCuda(
                    cudaEventRecord(kernelStart),
                    "kernel start record"
                );

                kickDrift<<<blocks, threads>>>(
                    dBodies,
                    dAccel,
                    DT,
                    NUM_BODIES
                );

                checkCuda(cudaGetLastError(), "kickDrift launch");

                computeAccelerations<<<blocks, threads, sharedBytes>>>(
                    dBodies,
                    dAccel,
                    NUM_BODIES
                );

                checkCuda(
                    cudaGetLastError(),
                    "computeAccelerations launch"
                );

                finalKick<<<blocks, threads>>>(
                    dBodies,
                    dAccel,
                    DT,
                    NUM_BODIES
                );

                checkCuda(cudaGetLastError(), "finalKick launch");

                checkCuda(
                    cudaEventRecord(kernelStop),
                    "kernel stop record"
                );

                checkCuda(
                    cudaEventSynchronize(kernelStop),
                    "kernel stop sync"
                );

                float stepMs = 0.0f;

                checkCuda(
                    cudaEventElapsedTime(
                        &stepMs,
                        kernelStart,
                        kernelStop
                    ),
                    "cudaEventElapsedTime"
                );

                totalKernelMs += stepMs;

                updateVboFromCuda(
                    cudaVbo,
                    dBodies,
                    blocks,
                    threads,
                    NUM_BODIES,
                    renderScale
                );

                checkCuda(
                    cudaMemcpy(
                        hBodies.data(),
                        dBodies,
                        NUM_BODIES * sizeof(Body),
                        cudaMemcpyDeviceToHost
                    ),
                    "history device-to-host copy"
                );

                ++newestStep;
                displayStep = newestStep;
                history.push_back(hBodies);
                stateChanged = true;

                if (newestStep % DIAGNOSTIC_INTERVAL == 0 ||
                    newestStep == NUM_STEPS) {
                    const Diagnostics current = calculateDiagnostics(
                        dBodies,
                        dDiagnostics,
                        blocks,
                        threads,
                        NUM_BODIES
                    );

                    printDiagnostics(
                        current,
                        initialDiagnostics,
                        newestStep
                    );
                }
            }

            if (displayStep >= NUM_STEPS) {
                gPlaying = false;
            }
        }

        if (stateChanged) {
            const double hours =
                static_cast<double>(displayStep) * DT / 3600.0;

            std::printf(
                "\r%s | Step %d / %d | %.3f hours      ",
                gPlaying ? "PLAYING" : "PAUSED",
                displayStep,
                NUM_STEPS,
                hours
            );

            std::fflush(stdout);
        }

        gPreviousSpace = spacePressed;
        gPreviousRight = rightPressed;
        gPreviousLeft = leftPressed;

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);

        if (height <= 0) {
            height = 1;
        }

        glViewport(0, 0, width, height);

        const float aspect =
            static_cast<float>(width) / static_cast<float>(height);

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram.id);

        glUniform1f(shaderProgram.yawLocation, gYaw);
        glUniform1f(shaderProgram.pitchLocation, gPitch);
        glUniform1f(shaderProgram.zoomLocation, gZoom);
        glUniform1f(shaderProgram.aspectLocation, aspect);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, NUM_BODIES);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    std::printf("\n\nCalculated steps: %d\n", newestStep);
    std::printf("Total leapfrog kernel time: %.3f ms\n", totalKernelMs);

    if (newestStep > 0) {
        std::printf(
            "Average physics step: %.6f ms\n",
            totalKernelMs / static_cast<float>(newestStep)
        );
    }

    checkCuda(
        cudaGraphicsUnregisterResource(cudaVbo),
        "cudaGraphicsUnregisterResource"
    );

    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    cudaFree(dDiagnostics);
    cudaFree(dAccel);
    cudaFree(dBodies);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    destroyShaderProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
