#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct GLFWwindow;

namespace atlas::interactive {

class Camera final {
public:
    Camera() = default;

    glm::mat4 view() const;
    glm::mat4 projection(float aspect_ratio) const;
    glm::mat4 view_projection(float aspect_ratio) const;
    void handle(GLFWwindow* window, float scroll_offset = 0.0f);

    void set_position(const glm::vec3& position) noexcept;
    void set_target(const glm::vec3& target) noexcept;
    void set_up(const glm::vec3& up) noexcept;
    void set_field_of_view(float degrees);
    void set_clip(float near_clip, float far_clip);

    const glm::vec3& position() const noexcept;
    const glm::vec3& target() const noexcept;
    float field_of_view() const noexcept;

private:
    void orbit(float yaw, float pitch);
    void zoom(float exponent);
    void set_axis_view(int index);
    void handle_view_shortcuts(GLFWwindow* window);
    void initialize_mouse(GLFWwindow* window);

    glm::vec3 _position { 0.0f, 0.0f, 3.0f };
    glm::vec3 _target { 0.0f, 0.0f, 0.0f };
    glm::vec3 _up { 0.0f, 1.0f, 0.0f };
    GLFWwindow* _window = nullptr;
    bool _left_drag_active = false;
    bool _right_drag_active = false;
    bool _view_key_down[7] { false, false, false, false, false, false, false };
    double _last_cursor_x = 0.0;
    double _last_cursor_y = 0.0;
    float _field_of_view = 45.0f;
    float _near_clip = 0.01f;
    float _far_clip = 1000.0f;
    float _minimum_distance = 1.0e-4f;
    float _maximum_distance = 1000.0f;
    float _orbit_sensitivity = 0.002f;
    float _drag_zoom_sensitivity = 0.0015f;
    float _scroll_zoom_sensitivity = 0.08f;
};

}
