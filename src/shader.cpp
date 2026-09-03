#include "shader.h"

#include <cstdio>
#include <cstdlib>

namespace
{
const char* vertexShaderSource = R"(
#version 330 core

layout (location = 0) in vec3 aPos;

uniform float yaw;
uniform float pitch;
uniform float zoom;
uniform float aspect;

void main()
{
    float cy = cos(yaw);
    float sy = sin(yaw);

    mat3 rotationY = mat3(
        cy,  0.0, -sy,
        0.0, 1.0,  0.0,
        sy,  0.0,  cy
    );

    float cp = cos(pitch);
    float sp = sin(pitch);

    mat3 rotationX = mat3(
        1.0, 0.0,  0.0,
        0.0,  cp,  -sp,
        0.0,  sp,   cp
    );

    vec3 position = rotationX * rotationY * aPos;
    position *= zoom;
    position.x /= aspect;

    gl_Position = vec4(position, 1.0);
    gl_PointSize = 3.0;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

void checkShaderCompile(unsigned int shader, const char* name)
{
    int success = 0;
    char infoLog[1024] = {};

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        fprintf(stderr, "%s shader compilation failed:\n%s\n", name, infoLog);
        exit(EXIT_FAILURE);
    }
}

void checkProgramLink(unsigned int program)
{
    int success = 0;
    char infoLog[1024] = {};

    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        fprintf(stderr, "Shader program linking failed:\n%s\n", infoLog);
        exit(EXIT_FAILURE);
    }
}
} // namespace

ShaderProgram createShaderProgram()
{
    ShaderProgram program;

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    checkShaderCompile(vertexShader, "Vertex");

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    checkShaderCompile(fragmentShader, "Fragment");

    program.id = glCreateProgram();
    glAttachShader(program.id, vertexShader);
    glAttachShader(program.id, fragmentShader);
    glLinkProgram(program.id);
    checkProgramLink(program.id);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    program.yawLocation = glGetUniformLocation(program.id, "yaw");
    program.pitchLocation = glGetUniformLocation(program.id, "pitch");
    program.zoomLocation = glGetUniformLocation(program.id, "zoom");
    program.aspectLocation = glGetUniformLocation(program.id, "aspect");

    return program;
}

void destroyShaderProgram(ShaderProgram program)
{
    if (program.id != 0)
    {
        glDeleteProgram(program.id);
    }
}
