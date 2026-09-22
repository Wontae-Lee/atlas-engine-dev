#include "rendering/backend/state_bridge.h"

#include "rendering/opengl/buffer.h"

namespace atlas::interactive {

struct StateBridge::Impl {};

StateBridge::StateBridge()
    : _impl(std::make_unique<Impl>()) {}

StateBridge::~StateBridge() = default;

void
StateBridge::release(opengl::Buffer&) {}

void
StateBridge::upload_raw(const void* source,
                        const std::size_t bytes,
                        opengl::Buffer& target) {
    target.upload(source, bytes);
    opengl::Buffer::unbind();
}

}
