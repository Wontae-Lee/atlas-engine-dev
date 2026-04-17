#pragma once

/**
 * @file host_buffer.h
 * @brief Defines the HostBuffer alias used for host-side buffer storage.
 */

#ifdef ATLAS_TASKING_CUDA
#include <thrust/host_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Buffer type used for host-side storage.
 *
 * This alias abstracts the underlying container used to store data in host
 * memory.
 *
 * Build configuration determines the actual type:
 * - when @c ATLAS_TASKING_CUDA is enabled, this resolves to @c thrust::host_vector<T>
 * - otherwise, it resolves to @c std::vector<T>
 *
 * This makes host-side code independent of the specific backend container type
 * while preserving a unified interface at the project level.
 *
 * @tparam T Element type stored in the buffer.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using HostBuffer = thrust::host_vector<T>;
#else
using HostBuffer = std::vector<T>;
#endif

} // namespace atlas