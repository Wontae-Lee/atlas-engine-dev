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
StateBridge::upload(const SimulationBufferView& source,
                    opengl::Buffer& target) {
    target.upload(source.data, source.bytes);
    opengl::Buffer::unbind();
}

}
