#pragma once

#include "rendering/opengl/framebuffer.h"
#include "rendering/target/render_target.h"

namespace atlas::interactive {

class OffscreenTarget final : public RenderTarget {
public:
    OffscreenTarget(int width, int height);

    void begin_frame() override;
    void end_frame() override;
    void present() override;
    int width() const noexcept override;
    int height() const noexcept override;
    bool should_close() const override;

    const opengl::Framebuffer& framebuffer() const noexcept;

private:
    opengl::Framebuffer _framebuffer;
    int _width = 0;
    int _height = 0;
};

}
