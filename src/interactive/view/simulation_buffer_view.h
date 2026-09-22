#pragma once

#include <cstddef>

namespace atlas::interactive {

struct SimulationBufferView final {
    const void* data = nullptr;
    std::size_t bytes = 0;
};

}
