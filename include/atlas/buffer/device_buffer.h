#pragma once

/**
 * @file device_buffer.h
 * @brief Declares the atlas::DeviceBuffer alias.
 *
 * This header defines `atlas::DeviceBuffer`, a backend-dependent container
 * alias used to store a contiguous sequence of elements.
 *
 * The selected container type depends on whether `ATLAS_TASKING_CUDA` is
 * defined at compile time.
 *
 * - If `ATLAS_TASKING_CUDA` is defined, the alias resolves to
 *   `thrust::device_vector<T>`.
 * - Otherwise, the alias resolves to `std::vector<T>`.
 */

#ifdef ATLAS_TASKING_CUDA
#include <thrust/device_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Backend-dependent buffer container alias.
 *
 * Defines a contiguous storage container whose concrete type is selected
 * at compile time based on the enabled tasking backend.
 *
 * When `ATLAS_TASKING_CUDA` is defined, this alias resolves to
 * `thrust::device_vector<T>`.
 *
 * When `ATLAS_TASKING_CUDA` is not defined, this alias resolves to
 * `std::vector<T>`.
 *
 * This alias allows code to refer to a single buffer type name without
 * directly depending on the backend-specific container type.
 *
 * @tparam T Element type stored in the buffer.
 *
 * @note The underlying container type, memory location, and access semantics
 * differ between CUDA-enabled and non-CUDA builds.
 *
 * @code
 * atlas::DeviceBuffer<float> buffer(256);
 * @endcode
 *
 * @see thrust::device_vector
 * @see std::vector
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using DeviceBuffer = thrust::device_vector<T>;
#else
using DeviceBuffer = std::vector<T>;
#endif

} // namespace atlas