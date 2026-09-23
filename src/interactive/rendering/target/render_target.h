/**
 * @file
 * @brief Declares the destination interface used by the renderer.
 */

#pragma once

namespace atlas::interactive {

/// Abstract destination and presentation lifecycle for OpenGL rendering.
class RenderTarget {
public:
    virtual ~RenderTarget() = default;
    /// Binds the target and prepares a new frame.
    virtual void begin_frame() = 0;
    /// Completes rendering commands for the current frame.
    virtual void end_frame() = 0;
    /// Makes the completed frame available to its consumer.
    virtual void present() = 0;
    /// Returns the current target width in pixels.
    virtual int width() const noexcept = 0;
    /// Returns the current target height in pixels.
    virtual int height() const noexcept = 0;
    /// Reports whether rendering should terminate.
    virtual bool should_close() const = 0;
};

}
