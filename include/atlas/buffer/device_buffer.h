#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/device_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @file device_buffer.h
 * @brief Backend-selected contiguous storage for "device" data.
 *
 * @details
 * This header defines `atlas::DeviceBuffer<T>` as an alias to a concrete container type
 * depending on the build configuration:
 *
 * - If `ATLAS_TASKING_CUDA` is defined:
 *   - `DeviceBuffer<T>` aliases `thrust::device_vector<T>`, which owns memory in CUDA device space
 *     and provides STL-like container semantics for device-resident data.
 *
 * - Otherwise:
 *   - `DeviceBuffer<T>` aliases `std::vector<T>`, providing a lightweight CPU fallback that
 *     preserves the same high-level API shape (size, resize, operator[], iterators, etc.).
 *
 * This allows higher-level systems to write code against `DeviceBuffer<T>` without
 * hard-wiring a particular backend in headers.
 *
 * @tparam T Element type stored in the buffer.
 *
 * @note
 * - The CPU fallback is primarily for testing and non-CUDA builds; it does not provide
 *   device execution or device pointers.
 * - `thrust::device_vector<T>` operations may incur device allocations and CUDA runtime calls.
 * - For best interoperability, prefer POD-like element types when storing data intended
 *   for device kernels.
 *
 * @see ATLAS_TASKING_CUDA
 */

/**
 * @brief Contiguous buffer type that maps to device memory in CUDA builds and host memory otherwise.
 *
 * @tparam T Element type.
 */
template <typename T>
#ifdef ATLAS_TASKING_CUDA
using DeviceBuffer = thrust::device_vector<T>;
#else
using DeviceBuffer = std::vector<T>;
#endif

} // namespace atlas
