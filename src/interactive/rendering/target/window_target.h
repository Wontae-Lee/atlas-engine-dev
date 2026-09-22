#pragma once

#include "rendering/target/render_target.h"

#include <string>

struct GLFWwindow;

namespace atlas::interactive {

class Camera;

class WindowTarget final : public RenderTarget {
public:
    WindowTarget(int width, int height, std::string title);
    ~WindowTarget() override;

    void begin_frame() override;
    void end_frame() override;
    void present() override;
    int width() const noexcept override;
    int height() const noexcept override;
    bool should_close() const override;
    void poll_events();
    void poll_events(Camera& camera);

private:
    static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);

    GLFWwindow* _window = nullptr;
    int _width = 0;
    int _height = 0;
    float _scroll_offset = 0.0f;
    std::string _title;
};

}
