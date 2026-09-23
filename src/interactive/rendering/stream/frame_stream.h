/**
 * @file
 * @brief Declares a sink for rendered offscreen frames.
 */

#pragma once

namespace atlas::interactive::opengl {
class Framebuffer;
}

namespace atlas::interactive {

/// Abstract consumer of frames produced by an offscreen renderer.
class FrameStream {
public:
    virtual ~FrameStream();

    /// Starts accepting frames.
    virtual void start() = 0;
    /// Stops accepting frames and releases stream resources.
    virtual void stop() = 0;
    /// Sends or encodes the current framebuffer contents.
    virtual void send(const opengl::Framebuffer& framebuffer) = 0;
};

}
