#pragma once

#include <atlas/core/macros.h>
#include <cstddef>
#include <cstdio>

#if defined(ATLAS_TASKING_CUDA)
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#elif defined(ATLAS_TASKING_TBB)
#include <cstring>
#else
#error "[ATLAS] Exactly one tasking backend must be selected: ATLAS_TASKING_CUDA or ATLAS_TASKING_TBB"
#endif

namespace atlas {

/**
 * @brief Copy a contiguous block of elements from device memory to host memory.
 *
 * @tparam T Element type being copied.
 *
 * @param src Pointer to the source memory region.
 * @param dst Pointer to the destination memory region.
 * @param count Number of elements to copy.
 *
 * @details
 * This helper provides a backend-unified interface for transferring data from
 * device-facing storage into host-facing storage.
 *
 * Backend behavior:
 * - Under `ATLAS_TASKING_CUDA`, this performs a `cudaMemcpy` with
 *   `cudaMemcpyDeviceToHost`.
 * - Under `ATLAS_TASKING_TBB`, device and host storage are treated as ordinary
 *   CPU-accessible memory, so the implementation falls back to `std::memcpy`.
 *
 * Error handling:
 * - CUDA copy failures are reported to `stderr`.
 * - The function does not throw and does not abort execution.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 *
 * @note
 * The caller is responsible for ensuring:
 * - `src` points to at least `count` readable elements,
 * - `dst` points to at least `count` writable elements,
 * - the source and destination memory regions do not overlap in unsupported ways.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    const std::size_t bytes = count * sizeof(T);

#if defined(ATLAS_TASKING_CUDA)
    const cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemcpy (device to host) failed: %s\n",
                     cudaGetErrorString(err));
    }
#elif defined(ATLAS_TASKING_TBB)
    std::memcpy(dst, src, bytes);
#endif
}

/**
 * @brief Copy a contiguous block of elements from host memory to device memory.
 *
 * @tparam T Element type being copied.
 *
 * @param src Pointer to the source memory region.
 * @param dst Pointer to the destination memory region.
 * @param count Number of elements to copy.
 *
 * @details
 * This helper provides a backend-unified interface for transferring data from
 * host-facing storage into device-facing storage.
 *
 * Backend behavior:
 * - Under `ATLAS_TASKING_CUDA`, this performs a `cudaMemcpy` with
 *   `cudaMemcpyHostToDevice`.
 * - Under `ATLAS_TASKING_TBB`, host and device storage are both ordinary
 *   CPU-accessible memory, so the implementation falls back to `std::memcpy`.
 *
 * Error handling:
 * - CUDA copy failures are reported to `stderr`.
 * - The function does not throw and does not abort execution.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 *
 * @note
 * The caller is responsible for ensuring:
 * - `src` points to at least `count` readable elements,
 * - `dst` points to at least `count` writable elements,
 * - the destination memory is valid for the currently selected backend.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    const std::size_t bytes = count * sizeof(T);

#if defined(ATLAS_TASKING_CUDA)
    const cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemcpy (host to device) failed: %s\n",
                     cudaGetErrorString(err));
    }
#elif defined(ATLAS_TASKING_TBB)
    std::memcpy(dst, src, bytes);
#endif
}

#if defined(ATLAS_TASKING_CUDA)

/**
 * @brief Copy a contiguous block of elements from host memory to a Thrust device pointer.
 *
 * @tparam T Element type being copied.
 *
 * @param src Pointer to the host source memory region.
 * @param dst Destination Thrust device pointer.
 * @param count Number of elements to copy.
 *
 * @details
 * This CUDA-only overload accepts a `thrust::device_ptr<T>` directly and copies
 * `count` elements from host memory into the pointed-to device allocation.
 *
 * Internally, the Thrust device pointer is converted into a raw CUDA pointer
 * using `thrust::raw_pointer_cast`, then transferred using `cudaMemcpy`.
 *
 * Error handling:
 * - CUDA copy failures are reported to `stderr`.
 * - The function does not throw.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, thrust::device_ptr<T> dst, const std::size_t count) {
    if (count == 0) return;

    const cudaError_t err = cudaMemcpy(thrust::raw_pointer_cast(dst),
                                       src,
                                       sizeof(T) * count,
                                       cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemcpy (host to device, thrust::device_ptr) failed: %s\n",
                     cudaGetErrorString(err));
    }
}

/**
 * @brief Copy a contiguous block of elements from host memory into a Thrust device vector.
 *
 * @tparam T Element type being copied.
 *
 * @param src Pointer to the host source memory region.
 * @param dst Destination Thrust device vector.
 * @param count Number of elements to copy.
 *
 * @details
 * This CUDA-only overload copies `count` elements from host memory into the
 * storage owned by a `thrust::device_vector<T>`.
 *
 * The copy is delegated to the `thrust::device_ptr<T>` overload by using
 * `dst.data()` as the destination pointer.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 *
 * @note
 * The caller is responsible for ensuring that `dst` has capacity for at least
 * `count` elements.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, thrust::device_vector<T>& dst, const std::size_t count) {
    if (count == 0) return;
    copy_host_to_device(src, dst.data(), count);
}

/**
 * @brief Copy a contiguous block of elements from a Thrust device pointer to host memory.
 *
 * @tparam T Element type being copied.
 *
 * @param src Source Thrust device pointer.
 * @param dst Pointer to the host destination memory region.
 * @param count Number of elements to copy.
 *
 * @details
 * This CUDA-only overload accepts a `thrust::device_ptr<const T>` as the source
 * and copies `count` elements into host memory.
 *
 * Internally, the Thrust device pointer is converted into a raw CUDA pointer
 * using `thrust::raw_pointer_cast`, then transferred using `cudaMemcpy`.
 *
 * Error handling:
 * - CUDA copy failures are reported to `stderr`.
 * - The function does not throw.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(thrust::device_ptr<const T> src, T* dst, const std::size_t count) {
    if (count == 0) return;

    const cudaError_t err = cudaMemcpy(dst,
                                       thrust::raw_pointer_cast(src),
                                       sizeof(T) * count,
                                       cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemcpy (device to host, thrust::device_ptr) failed: %s\n",
                     cudaGetErrorString(err));
    }
}

/**
 * @brief Copy a contiguous block of elements from a Thrust device vector to host memory.
 *
 * @tparam T Element type being copied.
 *
 * @param src Source Thrust device vector.
 * @param dst Pointer to the host destination memory region.
 * @param count Number of elements to copy.
 *
 * @details
 * This CUDA-only overload copies `count` elements from a
 * `thrust::device_vector<T>` into host memory.
 *
 * The implementation delegates to the `thrust::device_ptr<const T>` overload
 * by constructing a device pointer from `src.data()`.
 *
 * Special case:
 * - If `count == 0`, the function returns immediately and performs no action.
 *
 * @note
 * The caller is responsible for ensuring that `src` contains at least
 * `count` elements.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const thrust::device_vector<T>& src, T* dst, const std::size_t count) {
    if (count == 0) return;
    copy_device_to_host(thrust::device_ptr<const T>(src.data()), dst, count);
}

#endif

} // namespace atlas