#include <cuda_runtime.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nbody.cuh"
#include "shader.h"
#include "simulation.h"

float gYaw = 0.0f;
float gPitch = 0.0f;
float gZoom = 1.0f;

bool gDragging = false;

double gLastMouseX = 0.0;
double gLastMouseY = 0.0;

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    (void)window;
    (void)xOffset;

    const float zoomFactor = 1.15f;

    if (yOffset > 0.0)
        gZoom *= zoomFactor;
    else if (yOffset < 0.0)
        gZoom /= zoomFactor;

    gZoom = std::clamp(gZoom, 0.05f, 100.0f);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    (void)mods;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            gDragging = true;
            glfwGetCursorPos(window, &gLastMouseX, &gLastMouseY);
        }
        else if (action == GLFW_RELEASE)
        {
            gDragging = false;
        }
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    (void)window;

    if (!gDragging)
        return;

    double dx = xpos - gLastMouseX;
    double dy = ypos - gLastMouseY;

    gLastMouseX = xpos;
    gLastMouseY = ypos;

    const float sensitivity = 0.005f;
    gYaw += static_cast<float>(dx) * sensitivity;
    gPitch += static_cast<float>(dy) * sensitivity;
    gPitch = std::clamp(gPitch, -1.55f, 1.55f);
}

int main(int argc, char* argv[])
{
    const int N = 8192;
    const int steps = 1000;
    const float dt = 0.001f;

    int threads = 128;

    if (argc > 2)
    {
        printf("Usage: %s [threads_per_block]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2)
    {
        threads = std::atoi(argv[1]);
    }

    if (threads <= 0 || threads > 1024)
    {
        fprintf(stderr, "Error: threads_per_block must be between 1 and 1024.\n");
        return EXIT_FAILURE;
    }

    const int blocks = (N + threads - 1) / threads;
    const size_t sharedBytes = threads * sizeof(Body);

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1000, 800, "CUDA N-Body", nullptr, nullptr);
    if (!window)
    {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));

    glViewport(0, 0, 1000, 800);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_PROGRAM_POINT_SIZE);

    ShaderProgram shaderProgram = createShaderProgram();

    std::vector<Body> h_bodies(N);
    initializeBodies(h_bodies);

    Body* d_bodies = nullptr;
    float3* d_accel = nullptr;

    checkCuda(cudaMalloc(&d_bodies, N * sizeof(Body)), "cudaMalloc d_bodies");
    checkCuda(cudaMalloc(&d_accel, N * sizeof(float3)), "cudaMalloc d_accel");
    checkCuda(cudaMemcpy(d_bodies, h_bodies.data(), N * sizeof(Body), cudaMemcpyHostToDevice), "Copy bodies to GPU");

    printf("\n");
    printf("Bodies:              %d\n", N);
    printf("Threads/block:       %d\n", threads);
    printf("Blocks:              %d\n", blocks);
    printf("Shared memory/block: %zu bytes\n", sharedBytes);
    printf("Steps:               %d\n\n", steps);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");

    for (int step = 0; step < steps; ++step)
    {
        computeAccelerations<<<blocks, threads, sharedBytes>>>(d_bodies, d_accel, N);
        checkCuda(cudaGetLastError(), "computeAccelerations");

        integrate<<<blocks, threads>>>(d_bodies, d_accel, dt, N);
        checkCuda(cudaGetLastError(), "integrate");

        if (step % 100 == 0)
        {
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            printf("Step %d / %d\n", step, steps);
        }
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    printf("\nGPU time: %.3f ms\n", milliseconds);

    checkCuda(cudaMemcpy(h_bodies.data(), d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost), "Copy bodies from GPU");

    printf("\nFinal positions:\n\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("Body %4d : %10.6f %10.6f %10.6f\n",
               i,
               h_bodies[i].pos.x,
               h_bodies[i].pos.y,
               h_bodies[i].pos.z);
    }

    float maxRadius = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        float x = h_bodies[i].pos.x;
        float y = h_bodies[i].pos.y;
        float z = h_bodies[i].pos.z;
        float radius = std::sqrt(x * x + y * y + z * z);
        maxRadius = std::max(maxRadius, radius);
    }

    if (maxRadius < 1e-6f)
        maxRadius = 1.0f;

    const float renderMargin = 0.75f;
    const float renderScale = renderMargin / maxRadius;

    printf("\nMaximum radius: %.3f\n", maxRadius);
    printf("Render scale: %.8f\n", renderScale);

    std::vector<float> positions;
    positions.reserve(N * 3);
    for (int i = 0; i < N; ++i)
    {
        positions.push_back(h_bodies[i].pos.x * renderScale);
        positions.push_back(h_bodies[i].pos.y * renderScale);
        positions.push_back(h_bodies[i].pos.z * renderScale);
    }

    unsigned int VAO = 0;
    unsigned int VBO = 0;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_accel);
    cudaFree(d_bodies);

    printf("\nVisualization controls:\n");
    printf("  Left click + drag : rotate\n");
    printf("  Mouse wheel       : zoom\n");
    printf("  Close window      : exit\n\n");

    while (!glfwWindowShouldClose(window))
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (height == 0)
            height = 1;

        glViewport(0, 0, width, height);
        const float aspect = static_cast<float>(width) / static_cast<float>(height);

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram.id);
        glUniform1f(shaderProgram.yawLocation, gYaw);
        glUniform1f(shaderProgram.pitchLocation, gPitch);
        glUniform1f(shaderProgram.zoomLocation, gZoom);
        glUniform1f(shaderProgram.aspectLocation, aspect);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    destroyShaderProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}