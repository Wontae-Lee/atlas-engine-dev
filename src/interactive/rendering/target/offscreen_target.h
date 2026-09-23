/**
 * @file
 * @brief Declares framebuffer-backed rendering without presentation.
 */

#pragma once

#include "rendering/opengl/framebuffer.h"
#include "rendering/target/render_target.h"

namespace atlas::interactive {

/// Renders into an owned framebuffer without presenting to a window.
class OffscreenTarget final : public RenderTarget {
public:
    /// Creates an offscreen target of the supplied pixel dimensions.
    OffscreenTarget(int width, int height);

    void begin_frame() override;
    void end_frame() override;
    void present() override;
    int width() const noexcept override;
    int height() const noexcept override;
    bool should_close() const override;

    /// Returns the completed framebuffer for downstream frame consumers.
    const opengl::Framebuffer& framebuffer() const noexcept;

private:
    opengl::Framebuffer _framebuffer; ///< Framebuffer that receives rendered output.
    int _width = 0; ///< Requested width in pixels.
    int _height = 0; ///< Requested height in pixels.
};

}
