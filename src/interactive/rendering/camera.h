/**
 * @file
 * @brief Declares the interactive orbit camera.
 */

#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct GLFWwindow;

namespace atlas::interactive {

/// Orbit camera controlled through a GLFW window.
class Camera final {
public:
    Camera() = default;

    /// Returns the world-to-camera transform.
    glm::mat4 view() const;
    /// Returns the perspective projection for the supplied aspect ratio.
    glm::mat4 projection(float aspect_ratio) const;
    /// Returns the composed projection and view transform.
    glm::mat4 view_projection(float aspect_ratio) const;
    /// Applies keyboard, mouse-drag, and scroll input from a window.
    void handle(GLFWwindow* window, float scroll_offset = 0.0f);

    /// Sets the camera position in world coordinates.
    void set_position(const glm::vec3& position) noexcept;
    /// Sets the world-space orbit target.
    void set_target(const glm::vec3& target) noexcept;
    /// Sets the camera up direction.
    void set_up(const glm::vec3& up) noexcept;
    /// Sets the vertical field of view in degrees.
    void set_field_of_view(float degrees);
    /// Sets positive near and far clipping distances.
    void set_clip(float near_clip, float far_clip);

    /// Returns the world-space camera position.
    const glm::vec3& position() const noexcept;
    /// Returns the world-space orbit target.
    const glm::vec3& target() const noexcept;
    /// Returns the vertical field of view in degrees.
    float field_of_view() const noexcept;

private:
    void orbit(float yaw, float pitch);
    void zoom(float exponent);
    void set_axis_view(int index);
    void handle_view_shortcuts(GLFWwindow* window);
    void initialize_mouse(GLFWwindow* window);

    glm::vec3 _position { 0.0f, 0.0f, 3.0f }; ///< World-space eye position.
    glm::vec3 _target { 0.0f, 0.0f, 0.0f }; ///< World-space orbit target.
    glm::vec3 _up { 0.0f, 1.0f, 0.0f }; ///< Preferred camera up direction.
    GLFWwindow* _window = nullptr; ///< Window associated with cached cursor state.
    bool _left_drag_active = false; ///< Whether the previous sample was an orbit drag.
    bool _right_drag_active = false; ///< Whether the previous sample was a zoom drag.
    bool _view_key_down[7] { false, false, false, false, false, false, false }; ///< Shortcut edge state.
    double _last_cursor_x = 0.0; ///< Previous cursor x coordinate in screen pixels.
    double _last_cursor_y = 0.0; ///< Previous cursor y coordinate in screen pixels.
    float _field_of_view = 45.0f; ///< Vertical perspective field of view in degrees.
    float _near_clip = 0.01f; ///< Near clipping distance.
    float _far_clip = 1000.0f; ///< Far clipping distance.
    float _minimum_distance = 1.0e-4f; ///< Closest permitted orbit distance.
    float _maximum_distance = 1000.0f; ///< Furthest permitted orbit distance.
    float _orbit_sensitivity = 0.002f; ///< Radians of orbit per drag pixel.
    float _drag_zoom_sensitivity = 0.0015f; ///< Exponential zoom per drag pixel.
    float _scroll_zoom_sensitivity = 0.08f; ///< Exponential zoom per scroll unit.
};

}
