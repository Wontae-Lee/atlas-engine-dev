#pragma once

#include "rendering/target/render_target.h"

#include <string>

struct GLFWwindow;

namespace atlas::interactive {

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

private:
    GLFWwindow* _window = nullptr;
    int _width = 0;
    int _height = 0;
    std::string _title;
};

}
