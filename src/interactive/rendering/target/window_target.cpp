/**
 * @file
 * @brief Implements native-window and OpenGL-context ownership.
 */

#include "rendering/target/window_target.h"

#include "rendering/camera.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace atlas::interactive {

WindowTarget::WindowTarget(const int width, const int height, std::string title)
    : _width(width)
    , _height(height)
    , _title(std::move(title)) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Window dimensions must be positive.");
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed.");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(width, height, _title.c_str(), nullptr, nullptr);
    if (_window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed.");
    }

    glfwSetWindowUserPointer(_window, this);
    glfwSetScrollCallback(_window, &WindowTarget::scroll_callback);
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(0);
    // GLEW must be initialized after a context is current.
    glewExperimental = GL_TRUE;
    const GLenum result = glewInit();
    if (result != GLEW_OK) {
        glfwDestroyWindow(_window);
        _window = nullptr;
        glfwTerminate();
        throw std::runtime_error("GLEW initialization failed.");
    }
    // GLEW may emit a benign GL_INVALID_ENUM while probing a core-profile context.
    static_cast<void>(glGetError());
}

WindowTarget::~WindowTarget() {
    if (_window != nullptr) glfwDestroyWindow(_window);
    _window = nullptr;
    glfwTerminate();
}

void
WindowTarget::begin_frame() {
    glfwMakeContextCurrent(_window);
    glfwGetFramebufferSize(_window, &_width, &_height);
    _width = std::max(_width, 1);
    _height = std::max(_height, 1);
    glViewport(0, 0, _width, _height);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.025f, 0.035f, 0.055f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void
WindowTarget::end_frame() {
    glFlush();
}

void
WindowTarget::present() {
    glfwSwapBuffers(_window);
}

int
WindowTarget::width() const noexcept {
    return _width;
}

int
WindowTarget::height() const noexcept {
    return _height;
}

bool
WindowTarget::should_close() const {
    return glfwWindowShouldClose(_window) != 0;
}

void
WindowTarget::poll_events() {
    _scroll_offset = 0.0f;
    glfwPollEvents();
    if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(_window, GLFW_TRUE);
    }
}

void
WindowTarget::poll_events(Camera& camera) {
    poll_events();
    camera.handle(_window, _scroll_offset);
}

void
WindowTarget::scroll_callback(GLFWwindow* window, const double x_offset, const double y_offset) {
    static_cast<void>(x_offset);
    auto* target = static_cast<WindowTarget*>(glfwGetWindowUserPointer(window));
    if (target != nullptr) target->_scroll_offset += static_cast<float>(y_offset);
}

}
