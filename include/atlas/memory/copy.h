#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>

#include <cstddef>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace atlas {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(thrust::device_pointer_cast(src), count, dst);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(thrust::device_ptr<const T> src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const DeviceBuffer<T>& src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src.begin(), count, dst);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, thrust::device_pointer_cast(dst));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, thrust::device_ptr<T> dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, DeviceBuffer<T>& dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst.begin());
}

}