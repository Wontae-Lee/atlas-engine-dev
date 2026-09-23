/**
 * @file
 * @brief Declares a GLFW-backed native rendering target.
 */

#pragma once

#include "rendering/target/render_target.h"

#include <string>

struct GLFWwindow;

namespace atlas::interactive {

class Camera;

/// Owns a GLFW window, OpenGL context, and on-screen presentation.
class WindowTarget final : public RenderTarget {
public:
    /// Creates a visible native window and makes its context current.
    WindowTarget(int width, int height, std::string title);
    ~WindowTarget() override;

    /// @copydoc RenderTarget::begin_frame
    void begin_frame() override;
    /// @copydoc RenderTarget::end_frame
    void end_frame() override;
    /// @copydoc RenderTarget::present
    void present() override;
    /// @copydoc RenderTarget::width
    int width() const noexcept override;
    /// @copydoc RenderTarget::height
    int height() const noexcept override;
    /// @copydoc RenderTarget::should_close
    bool should_close() const override;
    /// Polls native events without applying camera controls.
    void poll_events();
    /// Polls native events and applies accumulated input to camera.
    void poll_events(Camera& camera);

private:
    static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);

    GLFWwindow* _window = nullptr; ///< Owned GLFW window handle.
    int _width = 0; ///< Current framebuffer width in pixels.
    int _height = 0; ///< Current framebuffer height in pixels.
    float _scroll_offset = 0.0f; ///< Scroll accumulated since the previous poll.
    std::string _title; ///< Native window title.
};

}
