#include <glad/glad.h>

#include "shader.h"

#include <cstdio>
#include <cstdlib>

namespace {

unsigned int compileShader(unsigned int type, const char* source) {
    const unsigned int shader = glCreateShader(type);

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        char log[4096]{};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);

        std::fprintf(stderr, "Shader compilation failed:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }

    return shader;
}

}  // namespace

ShaderProgram createShaderProgram() {
    static const char* vertexSource = R"(
        #version 330 core

        layout(location = 0) in vec3 aPosition;

        uniform float uYaw;
        uniform float uPitch;
        uniform float uZoom;
        uniform float uAspect;

        void main() {
            float cy = cos(uYaw);
            float sy = sin(uYaw);

            mat3 yawRotation = mat3(
                 cy, 0.0, -sy,
                0.0, 1.0, 0.0,
                 sy, 0.0,  cy
            );

            float cp = cos(uPitch);
            float sp = sin(uPitch);

            mat3 pitchRotation = mat3(
                1.0, 0.0, 0.0,
                0.0,  cp,  sp,
                0.0, -sp,  cp
            );

            vec3 p = pitchRotation * yawRotation * aPosition;
            p *= uZoom;

            if (uAspect > 1.0)
                p.x /= uAspect;
            else
                p.y *= uAspect;

            gl_Position = vec4(p, 1.0);
            gl_PointSize = 2.0;
        }
    )";

    static const char* fragmentSource = R"(
        #version 330 core

        out vec4 FragColor;

        void main() {
            FragColor = vec4(1.0);
        }
    )";

    const unsigned int vertexShader =
        compileShader(GL_VERTEX_SHADER, vertexSource);

    const unsigned int fragmentShader =
        compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    ShaderProgram program;
    program.id = glCreateProgram();

    glAttachShader(program.id, vertexShader);
    glAttachShader(program.id, fragmentShader);
    glLinkProgram(program.id);

    int success = 0;
    glGetProgramiv(program.id, GL_LINK_STATUS, &success);

    if (!success) {
        char log[4096]{};
        glGetProgramInfoLog(program.id, sizeof(log), nullptr, log);

        std::fprintf(stderr, "Shader link failed:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    program.yawLocation = glGetUniformLocation(program.id, "uYaw");
    program.pitchLocation = glGetUniformLocation(program.id, "uPitch");
    program.zoomLocation = glGetUniformLocation(program.id, "uZoom");
    program.aspectLocation = glGetUniformLocation(program.id, "uAspect");

    return program;
}

void destroyShaderProgram(const ShaderProgram& program) {
    if (program.id != 0) {
        glDeleteProgram(program.id);
    }
}
