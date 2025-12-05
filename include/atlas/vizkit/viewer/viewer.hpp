
#pragma once

#include <stdexcept>

namespace atlas::vizkit {

template <typename T>
Viewer<T>::Viewer(int width, int height, const char* title)
    : _width(width)
    , _height(height)
    , _title(title) { }

template <typename T>
Viewer<T>::~Viewer() {

    if (_win) {
        shutdown_layers();
        cleanup_gl();
    }
}

template <typename T>
void
Viewer<T>::add_layer(const std::shared_ptr<Layer<T>>& layer) {
    _layers.push_back(layer);
}

template <typename T>
int
Viewer<T>::run() {
    init_gl();
    init_layers();
    main_loop();
    shutdown_layers();
    cleanup_gl();
    return 0;
}

template <typename T>
void
Viewer<T>::init_gl() {
    if (!glfwInit()) {
        throw std::runtime_error("GLFW init failed");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _win = glfwCreateWindow(_width, _height, _title, nullptr, nullptr);
    if (!_win) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed");
    }

    glfwMakeContextCurrent(_win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwDestroyWindow(_win);
        _win = nullptr;
        glfwTerminate();
        throw std::runtime_error("GLEW init failed");
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
}

template <typename T>
void
Viewer<T>::init_layers() {
    for (auto& layer : _layers) {
        if (layer) {
            layer->init(_win, _cam);
        }
    }
}

template <typename T>
void
Viewer<T>::main_loop() {
    while (!glfwWindowShouldClose(_win)) {

        _cam.handle(_win);

        int w, h;
        glfwGetFramebufferSize(_win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (auto& layer : _layers) {
            if (layer) {
                layer->update(_win, _cam);
            }
        }

        glfwSwapBuffers(_win);
        glfwPollEvents();
    }
}

template <typename T>
void
Viewer<T>::shutdown_layers() {
    for (auto& layer : _layers) {
        if (layer) {
            layer->shutdown();
        }
    }
}

template <typename T>
void
Viewer<T>::cleanup_gl() {
    if (_win) {
        glfwDestroyWindow(_win);
        _win = nullptr;
    }
    glfwTerminate();
}

}
