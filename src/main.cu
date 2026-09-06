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


// ============================================================
// Camera state
// ============================================================

float gYaw = 0.0f;
float gPitch = 0.0f;
float gZoom = 1.0f;

bool gDragging = false;

double gLastMouseX = 0.0;
double gLastMouseY = 0.0;


// ============================================================
// Playback state
// ============================================================

bool gPlaying = false;


// ============================================================
// Keyboard edge detection
//
// We only want one step when an arrow key is PRESSED,
// not one step for every frame the key remains held.
// ============================================================

bool gPreviousSpace = false;
bool gPreviousRight = false;
bool gPreviousLeft = false;


// ============================================================
// Mouse wheel zoom
// ============================================================

void scrollCallback(
    GLFWwindow* window,
    double xOffset,
    double yOffset)
{
    (void)window;
    (void)xOffset;

    const float zoomFactor = 1.15f;

    if (yOffset > 0.0)
    {
        gZoom *= zoomFactor;
    }
    else if (yOffset < 0.0)
    {
        gZoom /= zoomFactor;
    }

    gZoom = std::clamp(
        gZoom,
        0.05f,
        100.0f
    );
}


// ============================================================
// Mouse button
// ============================================================

void mouseButtonCallback(
    GLFWwindow* window,
    int button,
    int action,
    int mods)
{
    (void)mods;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            gDragging = true;

            glfwGetCursorPos(
                window,
                &gLastMouseX,
                &gLastMouseY
            );
        }
        else if (action == GLFW_RELEASE)
        {
            gDragging = false;
        }
    }
}


// ============================================================
// Mouse drag rotation
// ============================================================

void cursorPositionCallback(
    GLFWwindow* window,
    double xpos,
    double ypos)
{
    (void)window;

    if (!gDragging)
    {
        return;
    }

    const double dx =
        xpos - gLastMouseX;

    const double dy =
        ypos - gLastMouseY;

    gLastMouseX = xpos;
    gLastMouseY = ypos;

    const float sensitivity = 0.005f;

    gYaw +=
        static_cast<float>(dx) *
        sensitivity;

    gPitch +=
        static_cast<float>(dy) *
        sensitivity;

    gPitch = std::clamp(
        gPitch,
        -1.55f,
        1.55f
    );
}


// ============================================================
// Convert Body positions into OpenGL positions
// ============================================================

void updateRenderPositions(
    const std::vector<Body>& bodies,
    std::vector<float>& positions,
    float renderScale)
{
    const size_t N =
        bodies.size();

    for (size_t i = 0;
         i < N;
         ++i)
    {
        positions[3 * i + 0] =
            bodies[i].pos.x *
            renderScale;

        positions[3 * i + 1] =
            bodies[i].pos.y *
            renderScale;

        positions[3 * i + 2] =
            bodies[i].pos.z *
            renderScale;
    }
}


// ============================================================
// Upload position array into OpenGL VBO
// ============================================================

void uploadPositions(
    unsigned int VBO,
    const std::vector<float>& positions)
{
    glBindBuffer(
        GL_ARRAY_BUFFER,
        VBO
    );

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        positions.size() *
            sizeof(float),
        positions.data()
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );
}


// ============================================================
// Main
// ============================================================

int main(
    int argc,
    char* argv[])
{
    const int N = 8192;
    const int steps = 1000;
    const float dt = 0.001f;

    int threads = 128;


    // ========================================================
    // Command-line arguments
    // ========================================================

    if (argc > 2)
    {
        printf(
            "Usage: %s [threads_per_block]\n",
            argv[0]
        );

        return EXIT_FAILURE;
    }

    if (argc == 2)
    {
        threads =
            std::atoi(
                argv[1]
            );
    }

    if (threads <= 0 ||
        threads > 1024)
    {
        fprintf(
            stderr,
            "Error: threads_per_block must be between 1 and 1024.\n"
        );

        return EXIT_FAILURE;
    }


    const int blocks =
        (N + threads - 1) /
        threads;

    const size_t sharedBytes =
        threads *
        sizeof(Body);


    // ========================================================
    // GLFW
    // ========================================================

    if (!glfwInit())
    {
        fprintf(
            stderr,
            "Failed to initialize GLFW\n"
        );

        return EXIT_FAILURE;
    }


    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        3
    );

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        3
    );

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );


    GLFWwindow* window =
        glfwCreateWindow(
            1000,
            800,
            "CUDA N-Body",
            nullptr,
            nullptr
        );


    if (!window)
    {
        fprintf(
            stderr,
            "Failed to create GLFW window\n"
        );

        glfwTerminate();

        return EXIT_FAILURE;
    }


    glfwMakeContextCurrent(
        window
    );


    // --------------------------------------------------------
    // VSync
    //
    // Playback will normally advance one simulation state
    // for each monitor refresh.
    // --------------------------------------------------------

    glfwSwapInterval(1);


    glfwSetScrollCallback(
        window,
        scrollCallback
    );

    glfwSetMouseButtonCallback(
        window,
        mouseButtonCallback
    );

    glfwSetCursorPosCallback(
        window,
        cursorPositionCallback
    );


    // ========================================================
    // GLAD
    // ========================================================

    if (!gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(
                glfwGetProcAddress
            )
        ))
    {
        fprintf(
            stderr,
            "Failed to initialize GLAD\n"
        );

        glfwDestroyWindow(
            window
        );

        glfwTerminate();

        return EXIT_FAILURE;
    }


    printf(
        "OpenGL version: %s\n",
        glGetString(GL_VERSION)
    );


    // ========================================================
    // OpenGL setup
    // ========================================================

    glViewport(
        0,
        0,
        1000,
        800
    );

    glClearColor(
        0.0f,
        0.0f,
        0.0f,
        1.0f
    );

    glEnable(
        GL_PROGRAM_POINT_SIZE
    );


    // ========================================================
    // Shader
    // ========================================================

    ShaderProgram shaderProgram =
        createShaderProgram();


    // ========================================================
    // Initial conditions
    // ========================================================

    std::vector<Body>
        h_bodies(N);

    initializeBodies(
        h_bodies
    );


    // ========================================================
    // Simulation history
    //
    // history[0] = initial conditions
    // history[1] = state after step 1
    // history[2] = state after step 2
    // ...
    //
    // This is what allows Left Arrow to rewind.
    // ========================================================

    std::vector<std::vector<Body>>
        history;

    history.reserve(
        steps + 1
    );

    history.push_back(
        h_bodies
    );


    // --------------------------------------------------------
    // displayStep:
    //
    // State currently being displayed.
    //
    // newestStep:
    //
    // Furthest timestep CUDA has actually calculated.
    // --------------------------------------------------------

    int displayStep = 0;

    int newestStep = 0;


    // ========================================================
    // CUDA memory
    // ========================================================

    Body* d_bodies =
        nullptr;

    float3* d_accel =
        nullptr;


    checkCuda(
        cudaMalloc(
            &d_bodies,
            N * sizeof(Body)
        ),
        "cudaMalloc d_bodies"
    );


    checkCuda(
        cudaMalloc(
            &d_accel,
            N * sizeof(float3)
        ),
        "cudaMalloc d_accel"
    );


    checkCuda(
        cudaMemcpy(
            d_bodies,
            h_bodies.data(),
            N * sizeof(Body),
            cudaMemcpyHostToDevice
        ),
        "Copy bodies to GPU"
    );


    // ========================================================
    // Configuration
    // ========================================================

    printf("\n");

    printf(
        "Bodies:              %d\n",
        N
    );

    printf(
        "Threads/block:       %d\n",
        threads
    );

    printf(
        "Blocks:              %d\n",
        blocks
    );

    printf(
        "Shared memory/block: %zu bytes\n",
        sharedBytes
    );

    printf(
        "Steps:               %d\n",
        steps
    );

    printf(
        "dt:                  %.6f\n\n",
        dt
    );


    // ========================================================
    // Render positions
    // ========================================================

    std::vector<float>
        positions(N * 3);


    // --------------------------------------------------------
    // Fixed rendering scale.
    //
    // We intentionally do NOT automatically rescale every
    // timestep because that would visually hide the physical
    // expansion/collapse of the system.
    // --------------------------------------------------------

    const float renderScale =
        0.75f;


    updateRenderPositions(
        h_bodies,
        positions,
        renderScale
    );


    // ========================================================
    // VAO / VBO
    // ========================================================

    unsigned int VAO = 0;

    unsigned int VBO = 0;


    glGenVertexArrays(
        1,
        &VAO
    );

    glGenBuffers(
        1,
        &VBO
    );


    glBindVertexArray(
        VAO
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        VBO
    );


    glBufferData(
        GL_ARRAY_BUFFER,
        positions.size() *
            sizeof(float),
        positions.data(),
        GL_DYNAMIC_DRAW
    );


    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        reinterpret_cast<void*>(0)
    );


    glEnableVertexAttribArray(
        0
    );


    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(
        0
    );


    // ========================================================
    // CUDA kernel timing
    //
    // We'll accumulate kernel execution time separately from
    // the visualization and CPU/GPU copy overhead.
    // ========================================================

    cudaEvent_t kernelStart =
        nullptr;

    cudaEvent_t kernelStop =
        nullptr;


    checkCuda(
        cudaEventCreate(
            &kernelStart
        ),
        "cudaEventCreate kernelStart"
    );


    checkCuda(
        cudaEventCreate(
            &kernelStop
        ),
        "cudaEventCreate kernelStop"
    );


    float totalKernelMilliseconds =
        0.0f;


    // ========================================================
    // Controls
    // ========================================================

    printf(
        "Visualization controls:\n"
    );

    printf(
        "  Space             : play / pause\n"
    );

    printf(
        "  Right Arrow       : one step forward\n"
    );

    printf(
        "  Left Arrow        : one step backward\n"
    );

    printf(
        "  Left click + drag : rotate\n"
    );

    printf(
        "  Mouse wheel       : zoom\n"
    );

    printf(
        "  ESC               : exit\n\n"
    );


    printf(
        "PAUSED | Step 0 / %d\n",
        steps
    );


    // ========================================================
    // Main loop
    // ========================================================

    while (!glfwWindowShouldClose(
               window))
    {
        // ====================================================
        // Process GLFW events first
        // ====================================================

        glfwPollEvents();


        // ====================================================
        // ESC
        // ====================================================

        if (glfwGetKey(
                window,
                GLFW_KEY_ESCAPE) ==
            GLFW_PRESS)
        {
            glfwSetWindowShouldClose(
                window,
                GLFW_TRUE
            );
        }


        // ====================================================
        // Read keyboard
        // ====================================================

        const bool spacePressed =
            glfwGetKey(
                window,
                GLFW_KEY_SPACE
            ) == GLFW_PRESS;


        const bool rightPressed =
            glfwGetKey(
                window,
                GLFW_KEY_RIGHT
            ) == GLFW_PRESS;


        const bool leftPressed =
            glfwGetKey(
                window,
                GLFW_KEY_LEFT
            ) == GLFW_PRESS;


        // ====================================================
        // Space = Play / Pause
        // ====================================================

        if (spacePressed &&
            !gPreviousSpace)
        {
            gPlaying =
                !gPlaying;

            printf(
                "\n%s | Step %d / %d\n",
                gPlaying
                    ? "PLAYING"
                    : "PAUSED",
                displayStep,
                steps
            );
        }


        // ====================================================
        // Left Arrow = rewind one timestep
        //
        // We automatically pause when manually stepping.
        // ====================================================

        bool stateChanged =
            false;


        if (leftPressed &&
            !gPreviousLeft)
        {
            gPlaying = false;

            if (displayStep > 0)
            {
                --displayStep;

                h_bodies =
                    history[displayStep];

                stateChanged =
                    true;


                printf(
                    "\rPAUSED | Step %d / %d      ",
                    displayStep,
                    steps
                );

                fflush(stdout);
            }
        }


        // ====================================================
        // Determine whether we should advance
        //
        // Right Arrow advances once.
        //
        // Playing advances once per rendered frame.
        // ====================================================

        bool advance =
            false;


        if (rightPressed &&
            !gPreviousRight)
        {
            gPlaying = false;

            advance = true;
        }


        if (gPlaying)
        {
            advance = true;
        }


        // ====================================================
        // Advance timestep
        // ====================================================

        if (advance &&
            displayStep < steps)
        {
            // ------------------------------------------------
            // CASE 1:
            //
            // We previously rewound.
            //
            // The requested next frame already exists in
            // history, so CUDA does NOT need to calculate it.
            // ------------------------------------------------

            if (displayStep < newestStep)
            {
                ++displayStep;

                h_bodies =
                    history[displayStep];

                stateChanged =
                    true;
            }


            // ------------------------------------------------
            // CASE 2:
            //
            // We're at the newest calculated timestep.
            //
            // CUDA must calculate a new state.
            // ------------------------------------------------

            else if (newestStep < steps)
            {
                // --------------------------------------------
                // CUDA currently contains newestStep.
                //
                // Start kernel timer.
                // --------------------------------------------

                checkCuda(
                    cudaEventRecord(
                        kernelStart
                    ),
                    "cudaEventRecord kernelStart"
                );


                // --------------------------------------------
                // Calculate acceleration
                // --------------------------------------------

                computeAccelerations
                    <<<blocks,
                       threads,
                       sharedBytes>>>(
                        d_bodies,
                        d_accel,
                        N
                    );


                checkCuda(
                    cudaGetLastError(),
                    "computeAccelerations"
                );


                // --------------------------------------------
                // Integrate
                // --------------------------------------------

                integrate
                    <<<blocks,
                       threads>>>(
                        d_bodies,
                        d_accel,
                        dt,
                        N
                    );


                checkCuda(
                    cudaGetLastError(),
                    "integrate"
                );


                // --------------------------------------------
                // Stop kernel timer
                // --------------------------------------------

                checkCuda(
                    cudaEventRecord(
                        kernelStop
                    ),
                    "cudaEventRecord kernelStop"
                );


                checkCuda(
                    cudaEventSynchronize(
                        kernelStop
                    ),
                    "cudaEventSynchronize kernelStop"
                );


                float stepMilliseconds =
                    0.0f;


                checkCuda(
                    cudaEventElapsedTime(
                        &stepMilliseconds,
                        kernelStart,
                        kernelStop
                    ),
                    "cudaEventElapsedTime"
                );


                totalKernelMilliseconds +=
                    stepMilliseconds;


                // --------------------------------------------
                // Retrieve the new state
                // --------------------------------------------

                checkCuda(
                    cudaMemcpy(
                        h_bodies.data(),
                        d_bodies,
                        N * sizeof(Body),
                        cudaMemcpyDeviceToHost
                    ),
                    "Copy bodies from GPU"
                );


                ++newestStep;

                displayStep =
                    newestStep;


                // --------------------------------------------
                // Store this state for rewind
                // --------------------------------------------

                history.push_back(
                    h_bodies
                );


                stateChanged =
                    true;
            }


            // ------------------------------------------------
            // Stop playing when the final timestep is reached.
            // ------------------------------------------------

            if (displayStep >= steps)
            {
                gPlaying =
                    false;
            }


            if (stateChanged)
            {
                printf(
                    "\r%s | Step %d / %d      ",
                    gPlaying
                        ? "PLAYING"
                        : "PAUSED",
                    displayStep,
                    steps
                );

                fflush(stdout);
            }
        }


        // ====================================================
        // Update OpenGL buffer only when timestep changes
        // ====================================================

        if (stateChanged)
        {
            updateRenderPositions(
                h_bodies,
                positions,
                renderScale
            );


            uploadPositions(
                VBO,
                positions
            );
        }


        // ====================================================
        // Save keyboard states
        // ====================================================

        gPreviousSpace =
            spacePressed;

        gPreviousRight =
            rightPressed;

        gPreviousLeft =
            leftPressed;


        // ====================================================
        // Window dimensions
        // ====================================================

        int width = 0;

        int height = 0;


        glfwGetFramebufferSize(
            window,
            &width,
            &height
        );


        if (height == 0)
        {
            height = 1;
        }


        glViewport(
            0,
            0,
            width,
            height
        );


        const float aspect =
            static_cast<float>(width) /
            static_cast<float>(height);


        // ====================================================
        // Draw current state
        // ====================================================

        glClear(
            GL_COLOR_BUFFER_BIT
        );


        glUseProgram(
            shaderProgram.id
        );


        glUniform1f(
            shaderProgram.yawLocation,
            gYaw
        );


        glUniform1f(
            shaderProgram.pitchLocation,
            gPitch
        );


        glUniform1f(
            shaderProgram.zoomLocation,
            gZoom
        );


        glUniform1f(
            shaderProgram.aspectLocation,
            aspect
        );


        glBindVertexArray(
            VAO
        );


        glDrawArrays(
            GL_POINTS,
            0,
            N
        );


        glfwSwapBuffers(
            window
        );
    }


    // ========================================================
    // Final information
    // ========================================================

    printf("\n\n");

    printf(
        "Displayed step:      %d\n",
        displayStep
    );

    printf(
        "Calculated steps:    %d\n",
        newestStep
    );

    printf(
        "CUDA kernel time:    %.3f ms\n",
        totalKernelMilliseconds
    );

    printf(
        "Stored snapshots:    %zu\n",
        history.size()
    );


    // ========================================================
    // Cleanup
    // ========================================================

    cudaEventDestroy(
        kernelStart
    );

    cudaEventDestroy(
        kernelStop
    );


    cudaFree(
        d_accel
    );

    cudaFree(
        d_bodies
    );


    glDeleteVertexArrays(
        1,
        &VAO
    );

    glDeleteBuffers(
        1,
        &VBO
    );


    destroyShaderProgram(
        shaderProgram
    );


    glfwDestroyWindow(
        window
    );

    glfwTerminate();


    return 0;
}