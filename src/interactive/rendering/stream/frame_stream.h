#pragma once

namespace atlas::interactive::opengl {
class Framebuffer;
}

namespace atlas::interactive {

class FrameStream {
public:
    virtual ~FrameStream();

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void send(const opengl::Framebuffer& framebuffer) = 0;
};

}
