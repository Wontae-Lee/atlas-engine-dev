#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>

#include <cstddef>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace atlas {

/**
 * @brief Copy @p count elements from a raw device address to a host buffer.
 *
 * The source is a bare @c T* holding a GPU address; it is wrapped in a
 * @c thrust::device_pointer_cast so thrust performs a device-to-host @c cudaMemcpy
 * rather than a host-side loop. All overloads here are @c ATLAS_HOST only — they issue
 * a synchronous transfer and must never be called from device code.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Raw device pointer to the first source element. May be anything when
 *            @p count is 0, in which case the call is a no-op.
 * @param dst Host destination; must have room for @p count elements.
 * @param count Number of elements to transfer; 0 short-circuits before touching @p src.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(thrust::device_pointer_cast(src), count, dst);
}

/**
 * @brief Copy @p count elements from a fancy device pointer to a host buffer.
 *
 * Same transfer as the raw-pointer overload, but the caller already holds a tagged
 * @c thrust::device_ptr<const T>, so no cast is needed.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Fancy device pointer to the first source element.
 * @param dst Host destination with room for @p count elements.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(thrust::device_ptr<const T> src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

/**
 * @brief Copy the first @p count elements of a @c DeviceBuffer to a host buffer.
 *
 * Convenience overload taking the owning container directly; reads from
 * @c src.begin(), which yields a device iterator, so thrust routes it as a
 * device-to-host transfer.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Device-resident source container; @p count must not exceed @c src.size().
 * @param dst Host destination with room for @p count elements.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(const DeviceBuffer<T>& src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src.begin(), count, dst);
}

/**
 * @brief Copy @p count elements from a host buffer to a raw device address.
 *
 * Mirror of @c copy_device_to_host: the destination bare @c T* holds a GPU address and
 * is wrapped in a @c thrust::device_pointer_cast so thrust issues a host-to-device
 * @c cudaMemcpy.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Host source with at least @p count elements.
 * @param dst Raw device pointer to the destination; may be anything when @p count is 0.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, thrust::device_pointer_cast(dst));
}

/**
 * @brief Copy @p count elements from a host buffer to a fancy device pointer.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Host source with at least @p count elements.
 * @param dst Fancy device pointer to the destination.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, thrust::device_ptr<T> dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

/**
 * @brief Copy @p count elements from a host buffer into a @c DeviceBuffer.
 *
 * Convenience overload writing into the owning container at @c dst.begin(); the caller
 * is responsible for having sized @p dst to hold at least @p count elements.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Host source with at least @p count elements.
 * @param dst Device-resident destination container, pre-sized to at least @p count.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, DeviceBuffer<T>& dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst.begin());
}

}