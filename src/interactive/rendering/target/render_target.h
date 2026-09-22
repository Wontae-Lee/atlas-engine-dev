#pragma once

namespace atlas::interactive {

class RenderTarget {
public:
    virtual ~RenderTarget() = default;
    virtual void begin_frame() = 0;
    virtual void end_frame() = 0;
    virtual void present() = 0;
    virtual int width() const noexcept = 0;
    virtual int height() const noexcept = 0;
    virtual bool should_close() const = 0;
};

}
