#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cstddef>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#else
#include <algorithm>
#endif

namespace atlas {

/**
 * @brief Copy @p count elements from a raw device address to a host buffer.
 *
 * Under the CUDA backend the source is wrapped in a @c thrust::device_pointer_cast so
 * thrust issues one device-to-host @c cudaMemcpy rather than a host-side loop; under the
 * host backend it is a @c std::copy_n. Every overload here is @c ATLAS_HOST only — it
 * issues a synchronous transfer and must never be called from device code.
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
#if defined(ATLAS_BACKEND_CUDA)
    thrust::copy_n(thrust::device_pointer_cast(src), count, dst);
#else
    std::copy_n(src, count, dst);
#endif
}

#if defined(ATLAS_BACKEND_CUDA)

/**
 * @brief Copy @p count elements from a fancy device pointer to a host buffer.
 *
 * Same transfer as the raw-pointer overload, but the caller already holds a tagged
 * @c device_ptr<const T>, so no cast is needed. Declared only under the CUDA backend,
 * where @c device_ptr is distinct from @c const @c T*.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Fancy device pointer to the first source element.
 * @param dst Host destination with room for @p count elements.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_device_to_host(device_ptr<const T> src, T* dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

#endif

/**
 * @brief Copy the first @p count elements of a @c DeviceBuffer to a host buffer.
 *
 * Convenience overload taking the owning container directly. Under CUDA, @c src.begin()
 * yields a device iterator, so thrust routes it as a device-to-host transfer.
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
#if defined(ATLAS_BACKEND_CUDA)
    thrust::copy_n(src.begin(), count, dst);
#else
    std::copy_n(src.begin(), count, dst);
#endif
}

/**
 * @brief Copy @p count elements from a host buffer to a raw device address.
 *
 * Mirror of @c copy_device_to_host: under CUDA the destination bare @c T* holds a GPU
 * address and is wrapped so thrust issues one host-to-device @c cudaMemcpy.
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
#if defined(ATLAS_BACKEND_CUDA)
    thrust::copy_n(src, count, thrust::device_pointer_cast(dst));
#else
    std::copy_n(src, count, dst);
#endif
}

#if defined(ATLAS_BACKEND_CUDA)

/**
 * @brief Copy @p count elements from a host buffer to a fancy device pointer.
 *
 * Declared only under the CUDA backend, where @c device_ptr is distinct from @c T*.
 *
 * @tparam T Element type; must be trivially copyable across the host/device boundary.
 * @param src Host source with at least @p count elements.
 * @param dst Fancy device pointer to the destination.
 * @param count Number of elements to transfer; 0 is a no-op.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
copy_host_to_device(const T* src, device_ptr<T> dst, const std::size_t count) {
    if (count == 0) return;
    thrust::copy_n(src, count, dst);
}

#endif

/**
 * @brief Copy @p count elements from a host buffer into a @c DeviceBuffer.
 *
 * Convenience overload writing into the owning container at @c dst.begin(); the caller is
 * responsible for having sized @p dst to hold at least @p count elements.
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
#if defined(ATLAS_BACKEND_CUDA)
    thrust::copy_n(src, count, dst.begin());
#else
    std::copy_n(src, count, dst.begin());
#endif
}

}
