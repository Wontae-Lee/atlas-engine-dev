#pragma once

/**
 * @file device_buffer.h
 * @brief Defines the DeviceBuffer alias used for device-oriented buffer storage.
 */

#ifdef ATLAS_TASKING_CUDA
#include <thrust/device_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Buffer type used for device-oriented storage.
 *
 * This alias abstracts the underlying container used to store data intended for
 * device-side or device-oriented computation.
 *
 * Build configuration determines the actual type:
 * - when @c ATLAS_TASKING_CUDA is enabled, this resolves to @c thrust::device_vector<T>
 * - otherwise, it resolves to @c std::vector<T>
 *
 * This indirection allows higher-level code to use a consistent buffer type
 * name regardless of whether CUDA support is enabled.
 *
 * @tparam T Element type stored in the buffer.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using DeviceBuffer = thrust::device_vector<T>;
#else
using DeviceBuffer = std::vector<T>;
#endif

} // namespace atlas