#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/host_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @file host_buffer.h
 * @brief Backend-selected contiguous storage for host-resident data.
 *
 * @details
 * This header defines `atlas::HostBuffer<T>` as a thin alias to a concrete container type
 * depending on the build configuration:
 *
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**
 *   `HostBuffer<T>` aliases `thrust::host_vector<T>`, which:
 *   - Allocates memory in host (CPU) space
 *   - Is interoperable with Thrust algorithms
 *   - Can be used as a staging buffer for transfers to/from `thrust::device_vector`
 *
 * - **Non-CUDA build**
 *   `HostBuffer<T>` aliases `std::vector<T>`, providing a lightweight and familiar
 *   CPU-only container with identical high-level semantics.
 *
 * The purpose of this alias is to allow higher-level code to depend on a single
 * host-side buffer abstraction, regardless of whether CUDA is enabled.
 *
 * Typical use cases:
 * - Owning host-resident data used to initialize or update device buffers
 * - CPU-side preprocessing, loading, or staging of simulation data
 * - Unified container type for algorithms that operate on host memory only
 *
 * @tparam T Element type stored in the buffer.
 *
 * @note
 * - `HostBuffer<T>` always resides in **host memory**, even in CUDA builds.
 * - In CUDA builds, prefer `HostBuffer<T>` over `std::vector<T>` when interacting
 *   with Thrust algorithms to avoid unnecessary conversions.
 * - The API surface intentionally matches common STL container operations
 *   (`size`, `resize`, `operator[]`, iterators).
 *
 * @see DeviceBuffer
 */

/**
 * @brief Contiguous buffer type for host-resident storage.
 *
 * @tparam T Element type.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using HostBuffer = thrust::host_vector<T>;
#else
using HostBuffer = std::vector<T>;
#endif

} // namespace atlas
