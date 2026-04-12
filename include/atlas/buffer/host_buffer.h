#pragma once

/**
 * @file host_buffer.h
 * @brief Declares the atlas::HostBuffer alias.
 *
 * This header defines `atlas::HostBuffer`, a backend-dependent container
 * alias for contiguous element storage in host-accessible memory.
 *
 * The concrete container type is selected at compile time:
 * - `thrust::host_vector<T>` when `ATLAS_TASKING_CUDA` is defined.
 * - `std::vector<T>` otherwise.
 */

#ifdef ATLAS_TASKING_CUDA
#include <thrust/host_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Backend-dependent contiguous host buffer container alias.
 *
 * `HostBuffer<T>` resolves to a backend-specific container type selected
 * at compile time.
 *
 * - Resolves to `thrust::host_vector<T>` when `ATLAS_TASKING_CUDA` is
 *   defined.
 * - Resolves to `std::vector<T>` otherwise.
 *
 * This alias provides a single host-side buffer type name that can be used
 * without directly depending on the backend-specific container type.
 *
 * @tparam T Element type stored in the buffer.
 *
 * @note The underlying container type may differ between CUDA-enabled and
 * non-CUDA builds, even though both variants provide host-accessible storage.
 *
 * @code
 * atlas::HostBuffer<int> buffer(1024);
 * @endcode
 *
 * @see thrust::host_vector
 * @see std::vector
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using HostBuffer = thrust::host_vector<T>;
#else
using HostBuffer = std::vector<T>;
#endif

} // namespace atlas