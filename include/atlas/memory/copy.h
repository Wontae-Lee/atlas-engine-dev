// <atlas/memory/copy.h>
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

/* =========================
 * Raw pointer copies
 * ========================= */

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

/* =========================
 * Thrust overloads (CUDA only)
 * ========================= */

#if defined(ATLAS_TASKING_CUDA)

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

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, thrust::device_vector<T>& dst, const std::size_t count) {
    if (count == 0) return;
    copy_host_to_device(src, dst.data(), count);
}

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

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const thrust::device_vector<T>& src, T* dst, const std::size_t count) {
    if (count == 0) return;
    copy_device_to_host(thrust::device_ptr<const T>(src.data()), dst, count);
}

#endif // ATLAS_TASKING_CUDA

} // namespace atlas
