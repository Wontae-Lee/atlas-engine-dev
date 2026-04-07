#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <string>
#include <utility>

namespace atlas::vizkit {

template <typename T>

Viewer<T>::Viewer(SystemHostPtr<T> system,
                  int width,
                  int height,
                  const char* title,
                  bool fullscreen) noexcept

    : _system(std::move(system))
    , _width(width)
    , _height(height)
    , _title(title ? title : "Atlas Viewer")
    , _fullscreen(fullscreen) { }

template <typename T>

Viewer<T>::~Viewer() {

    cleanup_gl();
}

template <typename T>
typename Viewer<T>::Builder
Viewer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Viewer<T>::add_layer(const std::shared_ptr<Layer<T>>& layer) {

    _layers.push_back(layer);
}

template <typename T>
const SystemHostPtr<T>&
Viewer<T>::system() const noexcept {
    return _system;
}

template <typename T>
int
Viewer<T>::run() {
    try {

        init_gl();
        init_layers();
        main_loop();
        shutdown_layers();
        cleanup_gl();
        return 0;
    } catch (const std::exception& e) {

        atlas::logger::error() << "Viewer failed: " << e.what();
        shutdown_layers();
        cleanup_gl();
        return 1;
    }
}

template <typename T>
void
Viewer<T>::init_gl() {

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        glfwTerminate();
        throw std::runtime_error("Failed to get primary monitor.");
    }

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!mode) {
        glfwTerminate();
        throw std::runtime_error("Failed to query primary monitor video mode.");
    }

    if (_width <= 0) _width = mode->width;
    if (_height <= 0) _height = mode->height;

    _win = glfwCreateWindow(
        _width,
        _height,
        _title,
        _fullscreen ? monitor : nullptr,
        nullptr);

    if (!_win) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    glfwMakeContextCurrent(_win);

    glewExperimental        = GL_TRUE;
    const GLenum glew_error = glewInit();
    if (glew_error != GLEW_OK) {
        glfwDestroyWindow(_win);
        _win = nullptr;
        glfwTerminate();

        throw std::runtime_error(
            std::string("Failed to initialize GLEW: ") + reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
    }

    glGetError();

    glEnable(GL_DEPTH_TEST);

    glLineWidth(2.0f);

    if (!_fullscreen) {

        const int x = (mode->width - _width) / 2;
        const int y = (mode->height - _height) / 2;
        glfwSetWindowPos(_win, x, y);
    }
}

template <typename T>
void
Viewer<T>::init_layers() {

    for (const auto& layer : _layers) {
        if (layer) {
            layer->init(_win, _cam);
        }
    }
}

template <typename T>
void
Viewer<T>::main_loop() {

    while (!glfwWindowShouldClose(_win)) {

        glfwPollEvents();

        if (glfwGetKey(_win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(_win, GLFW_TRUE);
        }

        _cam.handle(_win);

        int fbw = 1;
        int fbh = 1;
        glfwGetFramebufferSize(_win, &fbw, &fbh);

        glViewport(0, 0, fbw, fbh);

        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (const auto& layer : _layers) {
            if (layer) {
                layer->update(_win, _cam, _system->dt());
            }
        }

        glfwSwapBuffers(_win);
    }
}

template <typename T>
void
Viewer<T>::shutdown_layers() {

    for (const auto& layer : _layers) {
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

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_system(const SystemHostPtr<T>& system) {
    if (!system) {
        atlas::logger::error()
            << "Viewer::Builder: system must not be null.";
        throw std::runtime_error("Viewer::Builder: system must not be null.");
    }

    _system = system;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_title(const char* title) noexcept {

    _title = title ? title : "Atlas Viewer";
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_size(int width, int height) noexcept {

    _width  = width;
    _height = height;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_fullscreen(bool fullscreen) noexcept {
    _fullscreen = fullscreen;
    return *this;
}

template <typename T>
void
Viewer<T>::Builder::validate() const {
    if (_system == nullptr) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: system must not be null.";
        throw std::runtime_error("Viewer::Builder: system must not be null.");
    }

    if ((_width < 0) || (_height < 0)) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: width/height must be >= 0.";
        throw std::runtime_error("Viewer::Builder: width/height must be >= 0.");
    }

    if (_title == nullptr) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: title must not be null.";
        throw std::runtime_error("Viewer::Builder: title must not be null.");
    }
}

template <typename T>
Viewer<T>
Viewer<T>::Builder::build() const {

    validate();
    return Viewer<T>(_system, _width, _height, _title, _fullscreen);
}

template <typename T>
std::shared_ptr<Viewer<T>>
Viewer<T>::Builder::make_shared() const {

    auto viewer = build();
    return std::make_shared<Viewer<T>>(std::move(viewer));
}

}

#endif
