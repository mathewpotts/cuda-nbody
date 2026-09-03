#pragma once

#include <glad/glad.h>

struct ShaderProgram
{
    unsigned int id = 0;
    int yawLocation = -1;
    int pitchLocation = -1;
    int zoomLocation = -1;
    int aspectLocation = -1;
};

ShaderProgram createShaderProgram();
void destroyShaderProgram(ShaderProgram program);
