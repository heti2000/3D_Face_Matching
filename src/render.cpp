#include "./render.h"
#include <iostream>
#include <fstream>

const char* vertexShaderSource = R"(
    #version 330
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    layout(location = 2) in vec3 barycentric;
    layout(location = 3) in uint faceId;

    smooth out vec3 fragmentColor;
    smooth out vec3 fragmentBarycentric;
    flat out uint fragmentFaceId;

    // uniform mat4 projectionMatrix;
    
    void main() {
        // gl_Position = projectionMatrix * vec4(position, 1.0);
        gl_Position = vec4(position, 1.0);
        fragmentColor = color;
        fragmentBarycentric = barycentric;
        fragmentFaceId = faceId;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330
    smooth in vec3 fragmentColor;
    smooth in vec3 fragmentBarycentric;
    flat in uint fragmentFaceId;

    layout(location = 0) out vec3 pixelColor;
    layout(location = 1) out vec3 pixelBarycentric;
    layout(location = 2) out uint pixelFaceId;

    void main() {
        pixelColor = fragmentColor;
        pixelBarycentric = fragmentBarycentric;
        pixelFaceId = fragmentFaceId;
    }
)";

MeshRenderer::MeshRenderer(int width_, int height_): width(width_), height(height_) {
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit()){
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }

    /* Create a windowed mode window and its OpenGL context */
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // make the window invisible upon creation
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glViewport(0, 0, width, height);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Set background color to light blue
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST); // Enable depth testing
    glDepthFunc(GL_LEQUAL); // Specify depth testing function
    glDepthMask(GL_TRUE);
    glDepthRange(0.0, 1.0);
    
    glDisable(GL_CULL_FACE);

    glewExperimental = GL_TRUE;
    glewInit();

    // create textures for:
    // - the actual rendering
    // - the bayrcentric coordinates
    // - the corresponding face ids
    GLuint textureColor;
    glGenTextures(1, &textureColor);
    glBindTexture(GL_TEXTURE_2D, textureColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint textureBayrcentric;
    glGenTextures(1, &textureBayrcentric);
    glBindTexture(GL_TEXTURE_2D, textureBayrcentric);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint textureFaceId;
    glGenTextures(1, &textureFaceId);
    glBindTexture(GL_TEXTURE_2D, textureFaceId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // initialize buffers
    glGenBuffers(1, &shaderBuffer);
    glGenBuffers(1, &faceIdBuffer);

    // allocate memory
    glBindBuffer(GL_ARRAY_BUFFER, shaderBuffer);
    glBufferData(GL_ARRAY_BUFFER, (27 * BFM_N_FACES * sizeof(float)), NULL, GL_STATIC_DRAW);

    GLint size;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    
    glBindBuffer(GL_ARRAY_BUFFER, faceIdBuffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * BFM_N_FACES * sizeof(unsigned int), NULL, GL_STATIC_DRAW);

    // initialize and enable vertex array
    // // std::cout << "Loaded colors from buffer" << glGetError() << std::endl;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, shaderBuffer);

    // Load vertices, color and barycentric coordinates from the program
    glEnableVertexAttribArray(0);
    void* vertexPointer = 0;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(3*sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*sizeof(float), (void*)(6*sizeof(float)));

    // Load triangle ids from the separate buffer
    glBindBuffer(GL_ARRAY_BUFFER, faceIdBuffer);
    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, 0, (void*) 0);

    // unset VAO
    glBindVertexArray(0);

    // create a depthrenderbuffer for the renderer
    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);

    // create the frame buffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // attach the textures and depthrenderbuffer to the frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColor, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureBayrcentric, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, textureFaceId, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    // create the draw buffers
    GLenum DrawBuffers[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, DrawBuffers);

    GLenum framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE) {
        // std::cout << "-----ERROR-----" << std::endl;
        // std::cout << width << ", " << height << std::endl;
        // std::cout << "Framebuffer failed: Status " << framebufferStatus << std::endl;
        // std::cout << "---------------" << std::endl;
    }

    // Add shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint shaderStatus;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &shaderStatus);
    if (shaderStatus != GL_TRUE) {

        GLint maxLength = 0;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> errorLog(maxLength);
        glGetShaderInfoLog(vertexShader, maxLength, &maxLength, &errorLog[0]);

        std::string errorString(errorLog.begin(), errorLog.end());

        glDeleteShader(vertexShader);
        return;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &shaderStatus);
    if (shaderStatus != GL_TRUE) {
        
        GLint maxLength = 0;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> errorLog(maxLength);
        glGetShaderInfoLog(fragmentShader, maxLength, &maxLength, &errorLog[0]);

        std::string errorString(errorLog.begin(), errorLog.end());

        glDeleteShader(fragmentShader);
        return;
    }


    // Compile the shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDetachShader(shaderProgram, vertexShader);
    glDetachShader(shaderProgram, fragmentShader);
}


template <typename T>
void MeshRenderer::render(//const Eigen::Matrix<double,4,4>& projectionMatrix,
                const Eigen::Matrix<T,-1,1>& vertices, 
                const Eigen::Matrix<T,-1,1>& vertexColors,
                Eigen::VectorXi& triangles,
                float* pixelColor,
                float* pixelBarycentric,
                unsigned int* faceId) const {

    // init memory
    unsigned long nFaces = triangles.size() / 3;

    glBindBuffer(GL_ARRAY_BUFFER, shaderBuffer);

    GLint size;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);

    float* vertexData = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    for (int i=0; i<nFaces; i++) {
        for (int j=0; j<3; j++) {
            unsigned long vertIdx = triangles[i + j * nFaces];

            // vertex positions
            vertexData[9*(3*i+j)  ] = toFloat(vertices[3*vertIdx  ]);
            vertexData[9*(3*i+j)+1] = toFloat(vertices[3*vertIdx+1]);
            vertexData[9*(3*i+j)+2] = toFloat(vertices[3*vertIdx+2]);
            vertexData[9*(3*i+j)+3] = toFloat(vertexColors[3*vertIdx  ]);
            vertexData[9*(3*i+j)+4] = toFloat(vertexColors[3*vertIdx+1]);
            vertexData[9*(3*i+j)+5] = toFloat(vertexColors[3*vertIdx+2]);
            vertexData[9*(3*i+j)+6] = (float) (j % 3 == 0);
            vertexData[9*(3*i+j)+7] = (float) ((j-1) % 3 == 0);
            vertexData[9*(3*i+j)+8] = (float) ((j-2) % 3 == 0);
        }
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);

    unsigned int* faceIds = new unsigned int[nFaces * 3]; // 3 indices per triangle

    // Fill the faceIds array with triangle IDs (starting from 1)
    for (unsigned int i = 0; i < nFaces; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            faceIds[3 * i + j] = i + 1;
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, faceIdBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3*nFaces*sizeof(unsigned int), faceIds);

    //render the result
    glUseProgram(shaderProgram);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, faceIdBuffer);
    glBindVertexArray(vertexArray);

    glDrawArrays(GL_TRIANGLES, 0, triangles.size());

    // grab the rendering
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixelColor);

    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixelBarycentric);

    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, faceId);

    // unbind everything
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

template void MeshRenderer::render<ceres::Jet<double, 535>>(Eigen::Matrix<ceres::Jet<double, 535>, -1, 1, 0, -1, 1> const&, Eigen::Matrix<ceres::Jet<double, 535>, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1>&, float*, float*, unsigned int*) const;
template void MeshRenderer::render<ceres::Jet<double, 506>>(Eigen::Matrix<ceres::Jet<double, 506>, -1, 1, 0, -1, 1> const&, Eigen::Matrix<ceres::Jet<double, 506>, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1>&, float*, float*, unsigned int*) const;
template void MeshRenderer::render<double>(Eigen::Matrix<double, -1, 1, 0, -1, 1> const&, Eigen::Matrix<double, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1>&, float*, float*, unsigned int*) const;
template void MeshRenderer::render<float>(Eigen::Matrix<float, -1, 1, 0, -1, 1> const&, Eigen::Matrix<float, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1>&, float*, float*, unsigned int*) const;

std::ostream &operator<<(std::ostream &os, MeshRenderer const &m) { 
    return os << "MeshRenderer";
}