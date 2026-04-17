#pragma once

namespace atlas {

#ifdef ATLAS_TASKING_CUDA
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>

/**
 * @brief Backend-specific device pointer type alias for CUDA builds.
 *
 * @tparam T Pointee type.
 *
 * @details
 * Under CUDA tasking, Atlas uses `thrust::device_ptr<T>` as the canonical
 * device-pointer wrapper type. This provides compatibility with Thrust-based
 * containers and algorithms while still allowing explicit conversion to raw
 * pointers when needed.
 */
template <typename T>
using device_ptr = thrust::device_ptr<T>;

/**
 * @brief Return a raw pointer unchanged.
 *
 * @tparam T Pointee type.
 * @param p Raw mutable pointer.
 * @return The same raw pointer.
 *
 * @details
 * This overload exists so generic code can call `raw_pointer_cast(...)`
 * uniformly on both raw pointers and backend-specific pointer wrappers.
 *
 * For an already-raw pointer, no transformation is needed.
 */
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/**
 * @brief Return a raw const pointer unchanged.
 *
 * @tparam T Pointee type.
 * @param p Raw const pointer.
 * @return The same raw const pointer.
 *
 * @details
 * This overload mirrors the mutable raw-pointer case and allows generic
 * code to treat raw const pointers and wrapped device pointers through the
 * same function name.
 */
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

/**
 * @brief Convert a Thrust device pointer to its underlying raw pointer.
 *
 * @tparam T Pointee type.
 * @param p Thrust device pointer.
 * @return Raw mutable pointer to the same device memory.
 *
 * @details
 * This overload bridges Atlas generic code with Thrust pointer wrappers.
 * Internally it delegates to `thrust::raw_pointer_cast`.
 *
 * Marking the function `__host__ __device__` allows it to be used in both
 * host and device compilation contexts when supported by the underlying type.
 */
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

/**
 * @brief Convert a const Thrust device pointer to its underlying raw pointer.
 *
 * @tparam T Pointee type.
 * @param p Const Thrust device pointer.
 * @return Raw const pointer to the same device memory.
 *
 * @details
 * This overload provides the const-qualified companion to the mutable
 * `device_ptr<T>` conversion overload.
 */
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

#else

/**
 * @brief Backend-specific device pointer type alias for non-CUDA builds.
 *
 * @tparam T Pointee type.
 *
 * @details
 * In non-CUDA configurations, Atlas treats device pointers as ordinary raw
 * pointers. This keeps generic code portable without requiring a distinct
 * wrapper type on CPU-oriented backends.
 */
template <typename T>
using device_ptr = T*;

/**
 * @brief Return a raw pointer unchanged.
 *
 * @tparam T Pointee type.
 * @param p Raw mutable pointer.
 * @return The same raw pointer.
 *
 * @details
 * On non-CUDA backends, device memory is modeled directly with ordinary
 * pointers, so `raw_pointer_cast` is simply the identity operation.
 */
template <typename T>
constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/**
 * @brief Return a raw const pointer unchanged.
 *
 * @tparam T Pointee type.
 * @param p Raw const pointer.
 * @return The same raw const pointer.
 *
 * @details
 * Const-qualified identity overload for generic code that uses
 * `raw_pointer_cast(...)` regardless of backend.
 */
template <typename T>
constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

#endif

} // namespace atlas