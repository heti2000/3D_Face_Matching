#pragma once
#define GLFW_INCLUDE_GLU

#include <GL/glew.h>
#include <GLFW/glfw3.h>     
#include "constants.hpp"
#include "Eigen.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils/util.h"


class MeshRenderer {
public:
    MeshRenderer(int width, int height);
    
    template <typename T>
    void render(const Eigen::Matrix<T,-1,1>& vertices, 
                const Eigen::Matrix<T,-1,1>& vertexColors,
                Eigen::VectorXi& triangles,
                float* pixelColor,
                float* pixelBarycentric,
                unsigned int* faceId) const;

private:
    GLFWwindow* window;
    unsigned int fbo;

    int width, height;

    GLuint shaderProgram;

    GLuint shaderBuffer;
    GLuint faceIdBuffer;

    GLuint vertexArray;
};

std::ostream &operator<<(std::ostream &os, MeshRenderer const &m);
