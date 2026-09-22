#include "rendering/target/offscreen_target.h"

#include <stdexcept>

namespace atlas::interactive {

OffscreenTarget::OffscreenTarget(const int width, const int height)
    : _width(width)
    , _height(height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Offscreen dimensions must be positive.");
}

void
OffscreenTarget::begin_frame() {
    throw std::logic_error("OffscreenTarget does not yet own a headless OpenGL context.");
}

void
OffscreenTarget::end_frame() {}

void
OffscreenTarget::present() {}

int
OffscreenTarget::width() const noexcept {
    return _width;
}

int
OffscreenTarget::height() const noexcept {
    return _height;
}

bool
OffscreenTarget::should_close() const {
    return false;
}

const opengl::Framebuffer&
OffscreenTarget::framebuffer() const noexcept {
    return _framebuffer;
}

}
