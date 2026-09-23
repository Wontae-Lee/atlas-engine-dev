/**
 * @file
 * @brief Declares the backend-specific transfer from simulation buffers to OpenGL buffers.
 */

#pragma once

#include "view/simulation_buffer_view.h"

#include <memory>

namespace atlas::interactive::opengl {
class Buffer;
}

namespace atlas::interactive {

/**
 * @brief Copies backend-resident simulation bytes into reusable OpenGL buffers.
 *
 * The public interface is backend-independent. TBB builds upload CPU memory;
 * CUDA builds use registered graphics resources and device-to-device copies.
 */
class StateBridge final {
public:
    /// Creates backend-specific transfer state.
    StateBridge();
    StateBridge(const StateBridge&) = delete;
    StateBridge(StateBridge&&) = delete;
    /// Releases registrations and backend resources.
    ~StateBridge();

    StateBridge& operator=(const StateBridge&) = delete;
    StateBridge& operator=(StateBridge&&) = delete;

    /**
     * @brief Uploads a simulation buffer and sets the target's logical size.
     * @param source Non-owning view whose address belongs to the active backend.
     * @param target Persistent OpenGL destination buffer.
     */
    void upload(const SimulationBufferView& source, opengl::Buffer& target);

    /// Releases backend registration associated with a target buffer.
    void release(opengl::Buffer& target);

private:
    struct Impl;

    std::unique_ptr<Impl> _impl; ///< Backend-specific registration state.
};

}
