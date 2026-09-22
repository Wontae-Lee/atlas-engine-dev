#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace atlas::interactive {

class Camera final {
public:
    Camera() = default;

    glm::mat4 view() const;
    glm::mat4 projection(float aspect_ratio) const;
    glm::mat4 view_projection(float aspect_ratio) const;

    void set_position(const glm::vec3& position) noexcept;
    void set_target(const glm::vec3& target) noexcept;
    void set_up(const glm::vec3& up) noexcept;
    void set_field_of_view(float degrees);
    void set_clip(float near_clip, float far_clip);

    const glm::vec3& position() const noexcept;
    const glm::vec3& target() const noexcept;
    float field_of_view() const noexcept;

private:
    glm::vec3 _position { 0.0f, 0.0f, 3.0f };
    glm::vec3 _target { 0.0f, 0.0f, 0.0f };
    glm::vec3 _up { 0.0f, 1.0f, 0.0f };
    float _field_of_view = 45.0f;
    float _near_clip = 0.01f;
    float _far_clip = 1000.0f;
};

}
