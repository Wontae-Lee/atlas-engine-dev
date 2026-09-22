#include "rendering/camera.h"

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace atlas::interactive {

glm::mat4
Camera::view() const {
    return glm::lookAt(_position, _target, _up);
}

glm::mat4
Camera::projection(const float aspect_ratio) const {
    if (aspect_ratio <= 0.0f) throw std::invalid_argument("Camera aspect ratio must be positive.");
    return glm::perspective(glm::radians(_field_of_view), aspect_ratio, _near_clip, _far_clip);
}

glm::mat4
Camera::view_projection(const float aspect_ratio) const {
    return projection(aspect_ratio) * view();
}

void
Camera::handle(GLFWwindow* window, const float scroll_offset) {
    if (window == nullptr) return;
    initialize_mouse(window);

    double cursor_x = _last_cursor_x;
    double cursor_y = _last_cursor_y;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);
    const float delta_x = static_cast<float>(cursor_x - _last_cursor_x);
    const float delta_y = static_cast<float>(cursor_y - _last_cursor_y);

    const bool left_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (left_pressed && _left_drag_active) {
        orbit(-delta_x * _orbit_sensitivity, -delta_y * _orbit_sensitivity);
    }
    _left_drag_active = left_pressed;

    const bool right_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (right_pressed && _right_drag_active) zoom(delta_y * _drag_zoom_sensitivity);
    _right_drag_active = right_pressed;

    _last_cursor_x = cursor_x;
    _last_cursor_y = cursor_y;

    handle_view_shortcuts(window);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) orbit(-0.02f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) orbit(0.02f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) orbit(0.0f, 0.02f);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) orbit(0.0f, -0.02f);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) zoom(std::log(1.01f));
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) zoom(std::log(0.99f));

    zoom(-scroll_offset * _scroll_zoom_sensitivity);
}

void
Camera::set_position(const glm::vec3& position) noexcept {
    _position = position;
}

void
Camera::set_target(const glm::vec3& target) noexcept {
    _target = target;
}

void
Camera::set_up(const glm::vec3& up) noexcept {
    _up = up;
}

void
Camera::set_field_of_view(const float degrees) {
    if (degrees <= 0.0f || degrees >= 180.0f) {
        throw std::invalid_argument("Camera field of view must be between 0 and 180 degrees.");
    }
    _field_of_view = degrees;
}

void
Camera::set_clip(const float near_clip, const float far_clip) {
    if (near_clip <= 0.0f || far_clip <= near_clip) {
        throw std::invalid_argument("Camera clip planes are invalid.");
    }
    _near_clip = near_clip;
    _far_clip = far_clip;
}

const glm::vec3&
Camera::position() const noexcept {
    return _position;
}

const glm::vec3&
Camera::target() const noexcept {
    return _target;
}

float
Camera::field_of_view() const noexcept {
    return _field_of_view;
}

void
Camera::orbit(const float yaw, const float pitch) {
    glm::vec3 offset = _position - _target;
    const glm::vec3 up = glm::normalize(_up);
    offset = glm::vec3(glm::rotate(glm::mat4(1.0f), yaw, up) * glm::vec4(offset, 0.0f));

    const glm::vec3 direction = glm::normalize(-offset);
    const glm::vec3 pitch_axis = glm::normalize(glm::cross(up, direction));
    const glm::vec3 pitched = glm::vec3(
        glm::rotate(glm::mat4(1.0f), pitch, pitch_axis) * glm::vec4(offset, 0.0f));
    if (std::abs(glm::dot(glm::normalize(-pitched), up)) < 0.9999f) offset = pitched;

    _position = _target + offset;
}

void
Camera::zoom(const float exponent) {
    const glm::vec3 offset = _position - _target;
    const float distance = glm::length(offset);
    const float next_distance = std::clamp(
        distance * std::exp(exponent), _minimum_distance, _maximum_distance);
    _position = _target + offset * (next_distance / distance);
}

void
Camera::set_axis_view(const int index) {
    const float distance = glm::length(_position - _target);
    switch (index) {
    case 0:
        _position = _target + glm::vec3(0.0f, 0.0f, distance);
        _up = { 0.0f, 1.0f, 0.0f };
        break;
    case 1:
        _position = _target + glm::vec3(0.0f, 0.0f, -distance);
        _up = { 0.0f, 1.0f, 0.0f };
        break;
    case 2:
        _position = _target + glm::vec3(0.0f, distance, 0.0f);
        _up = { 0.0f, 0.0f, -1.0f };
        break;
    case 3:
        _position = _target + glm::vec3(0.0f, -distance, 0.0f);
        _up = { 0.0f, 0.0f, 1.0f };
        break;
    case 4:
        _position = _target + glm::vec3(-distance, 0.0f, 0.0f);
        _up = { 0.0f, 1.0f, 0.0f };
        break;
    case 5:
        _position = _target + glm::vec3(distance, 0.0f, 0.0f);
        _up = { 0.0f, 1.0f, 0.0f };
        break;
    case 6:
        _position = _target + glm::normalize(glm::vec3(1.0f, 0.75f, 1.0f)) * distance;
        _up = { 0.0f, 1.0f, 0.0f };
        break;
    default:
        break;
    }
}

void
Camera::handle_view_shortcuts(GLFWwindow* window) {
    constexpr int keys[7] = {
        GLFW_KEY_1,
        GLFW_KEY_2,
        GLFW_KEY_3,
        GLFW_KEY_4,
        GLFW_KEY_5,
        GLFW_KEY_6,
        GLFW_KEY_7,
    };
    for (int index = 0; index < 7; ++index) {
        const bool pressed = glfwGetKey(window, keys[index]) == GLFW_PRESS;
        if (pressed && !_view_key_down[index]) set_axis_view(index);
        _view_key_down[index] = pressed;
    }
}

void
Camera::initialize_mouse(GLFWwindow* window) {
    if (_window == window) return;
    _window = window;
    glfwGetCursorPos(window, &_last_cursor_x, &_last_cursor_y);
    _left_drag_active = false;
    _right_drag_active = false;
}

}
