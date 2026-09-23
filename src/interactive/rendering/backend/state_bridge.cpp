/**
 * @file
 * @brief Implements CPU simulation-buffer uploads to OpenGL.
 */

#include "rendering/backend/state_bridge.h"

#include "rendering/opengl/buffer.h"

namespace atlas::interactive {

/// Empty CPU-backend implementation state retained for a uniform ABI.
struct StateBridge::Impl {};

StateBridge::StateBridge()
    : _impl(std::make_unique<Impl>()) {}

StateBridge::~StateBridge() = default;

void
StateBridge::release(opengl::Buffer&) {}

void
StateBridge::upload(const SimulationBufferView& source,
                    opengl::Buffer& target) {
    // TBB DeviceBuffer storage is host-accessible, so OpenGL can consume it directly.
    target.upload(source.data, source.bytes);
    opengl::Buffer::unbind();
}

}
