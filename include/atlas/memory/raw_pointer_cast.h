#pragma once

#include <atlas/core/macros.h>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/device_ptr.h>
#endif

namespace atlas {

/**
 * @brief A pointer into the parallel backend's memory.
 *
 * Under the CUDA backend this is @c thrust::device_ptr<T>, a tagged pointer that thrust
 * routes through @c cudaMemcpy when it is dereferenced on the host. Under the host
 * backend there is no separate memory space, so it is a plain @c T*.
 *
 * @tparam T Pointee type.
 */
#if defined(ATLAS_BACKEND_CUDA)
template <typename T>
using device_ptr = thrust::device_ptr<T>;
#else
template <typename T>
using device_ptr = T*;
#endif

/**
 * @brief Strip any fancy-pointer wrapper and return the underlying raw address.
 *
 * The raw-pointer overloads are the identity; they exist so a caller can write
 * @c raw_pointer_cast(buffer.data()) without knowing which backend supplied the pointer.
 *
 * @tparam T Pointee type.
 * @param p Pointer to unwrap; may be null.
 * @return The raw address @p p refers to.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/** @copydoc raw_pointer_cast(T*) */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

#if defined(ATLAS_BACKEND_CUDA)

/**
 * @brief Unwrap a @c thrust::device_ptr into the bare device address it holds.
 *
 * Declared only under the CUDA backend. The host backend aliases @c device_ptr<T> to
 * @c T*, which would make these redeclarations of the raw-pointer overloads above.
 *
 * @tparam T Pointee type.
 * @param p Fancy device pointer.
 * @return The raw device address: safe to hand to a kernel, not to dereference on the
 *         host.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

/** @copydoc raw_pointer_cast(device_ptr<T>) */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

#endif

}
