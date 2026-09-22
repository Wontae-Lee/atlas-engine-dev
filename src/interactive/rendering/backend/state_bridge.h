#pragma once

#include "view/simulation_buffer_view.h"

#include <memory>

namespace atlas::interactive::opengl {
class Buffer;
}

namespace atlas::interactive {

class StateBridge final {
public:
    StateBridge();
    StateBridge(const StateBridge&) = delete;
    StateBridge(StateBridge&&) = delete;
    ~StateBridge();

    StateBridge& operator=(const StateBridge&) = delete;
    StateBridge& operator=(StateBridge&&) = delete;

    void upload(const SimulationBufferView& source, opengl::Buffer& target);
    void release(opengl::Buffer& target);

private:
    struct Impl;

    std::unique_ptr<Impl> _impl;
};

}
