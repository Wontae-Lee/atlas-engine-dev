#pragma once

/**
 * @file raw_pointer_cast.h
 * @brief Lightweight device pointer abstraction and raw pointer extraction helper.
 *
 * @details
 * This header provides a small compatibility layer so higher-level code can use a
 * single “device pointer” type and a single `raw_pointer_cast(...)` function across
 * CUDA and non-CUDA builds.
 *
 * Build modes:
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - `atlas::device_ptr<T>` is `thrust::device_ptr<T>`.
 *   - `raw_pointer_cast(...)` forwards to `thrust::raw_pointer_cast(...)` for device_ptr,
 *     and acts as an identity for raw pointers.
 *
 * - **CPU-only build**:
 *   - `atlas::device_ptr<T>` is simply `T*`.
 *   - `raw_pointer_cast(...)` is an identity function.
 *
 * This is especially useful when writing code that stores pointers in containers that
 * differ between backends (e.g., `thrust::device_vector` vs `std::vector`) and you still
 * need to obtain a raw pointer for low-level APIs or kernels.
 *
 * @note
 * The overload set intentionally supports both `T*` and `device_ptr<T>` so callers can
 * use `raw_pointer_cast(...)` without knowing the underlying pointer wrapper type.
 */

namespace atlas {

#ifdef ATLAS_TASKING_CUDA
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>

/**
 * @brief Device pointer wrapper type for CUDA builds.
 */
template <typename T>
using device_ptr = thrust::device_ptr<T>;

/**
 * @brief Identity overload for raw pointers (non-owning).
 *
 * @tparam T Element type.
 * @param p Raw pointer.
 * @return Same pointer `p`.
 */
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/**
 * @brief Identity overload for const raw pointers (non-owning).
 *
 * @tparam T Element type.
 * @param p Const raw pointer.
 * @return Same pointer `p`.
 */
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

/**
 * @brief Extracts the raw pointer from a `thrust::device_ptr<T>`.
 *
 * @tparam T Element type.
 * @param p Thrust device pointer wrapper.
 * @return Raw pointer suitable for kernel launches / CUDA APIs.
 *
 * @see thrust::raw_pointer_cast
 */
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

/**
 * @brief Extracts the raw pointer from a `thrust::device_ptr<const T>`.
 *
 * @tparam T Element type.
 * @param p Thrust const device pointer wrapper.
 * @return Raw const pointer.
 *
 * @see thrust::raw_pointer_cast
 */
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

#else // -----------------------------------------------------------------------
// CPU-only build
// -----------------------------------------------------------------------

/**
 * @brief Device pointer type for non-CUDA builds.
 *
 * @details
 * In CPU-only builds, "device pointers" are treated as ordinary raw pointers.
 */
template <typename T>
using device_ptr = T*;

/**
 * @brief Identity overload for raw pointers (CPU-only build).
 */
template <typename T>
constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

/**
 * @brief Identity overload for const raw pointers (CPU-only build).
 */
template <typename T>
constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}



#endif // ATLAS_TASKING_CUDA

} // namespace atlas
