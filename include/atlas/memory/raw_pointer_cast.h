#pragma once

#include <atlas/core/macros.h>

#include <thrust/device_ptr.h>

namespace atlas {

/**
 * @brief Alias for a fancy (tagged) pointer into device memory.
 *
 * @c thrust::device_ptr<T> carries the "this address lives on the GPU" tag that lets
 * thrust route dereferences and algorithms to the device system. The overloads below
 * strip that tag back down to a bare @c T* for handing to a kernel or another raw API.
 *
 * @tparam T Pointee type.
 */
template <typename T>
using device_ptr = thrust::device_ptr<T>;

/**
 * @brief Identity overload: a raw pointer is already raw, so return it unchanged.
 *
 * Present so generic code can call @c raw_pointer_cast uniformly on either a bare
 * pointer or a @c device_ptr without knowing which it holds.
 *
 * @tparam T Pointee type.
 * @param p Any pointer (host or device address); ownership and validity are unchanged.
 * @return @p p verbatim.
 * @note @c ATLAS_ALL_DEVICE + @c constexpr: usable in host code, device code, and
 *       constant expressions alike.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/**
 * @brief Const-qualified identity overload of @c raw_pointer_cast.
 *
 * @tparam T Pointee type (the pointer is to @c const @p T).
 * @param p Any pointer to const; returned verbatim.
 * @return @p p verbatim.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

/**
 * @brief Extract the underlying bare device address from a @c device_ptr<T>.
 *
 * Delegates to @c thrust::raw_pointer_cast. The result is a plain @c T* holding a GPU
 * address; it is only legal to dereference inside device code (or to pass to a kernel /
 * device-side API), never on the host.
 *
 * @tparam T Pointee type.
 * @param p Fancy device pointer to unwrap.
 * @return The raw device address @p p refers to.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

/**
 * @brief Const-qualified counterpart: unwrap a @c device_ptr<const T> to @c const T*.
 *
 * @tparam T Pointee type (the fancy pointer targets @c const @p T).
 * @param p Fancy device pointer to const to unwrap.
 * @return The raw device address, as a pointer to const.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

}