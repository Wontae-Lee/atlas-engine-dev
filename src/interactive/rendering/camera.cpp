#include "rendering/camera.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

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

}
