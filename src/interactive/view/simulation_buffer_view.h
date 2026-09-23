/**
 * @file
 * @brief Defines a non-owning byte view of a simulation buffer.
 */

#pragma once

#include <cstddef>

namespace atlas::interactive {

/**
 * @brief Non-owning untyped view of contiguous backend-resident bytes.
 *
 * The address may refer to CPU or CUDA memory according to the compiled Atlas
 * backend and remains valid only while the owning simulation buffer is stable.
 */
struct SimulationBufferView final {
    const void* data = nullptr; ///< Backend address of the first byte.
    std::size_t bytes = 0; ///< Number of live bytes available from data.
};

}
