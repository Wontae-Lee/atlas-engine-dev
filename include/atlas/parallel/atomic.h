#pragma once

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief Atomically adds @p value to `*address` and returns the prior value.
 *
 * A single portable spelling of an atomic fetch-and-add usable from both host
 * and device code. The implementation is chosen by whether the compiler is
 * currently generating device code:
 *   - In a `__CUDA_ARCH__` (device) compilation pass it forwards to CUDA's
 *     `atomicAdd`, whose overloads determine which types are supported on the
 *     GPU (integer types and the floating-point types the target architecture
 *     provides).
 *   - Otherwise it uses the GCC/Clang `__atomic_fetch_add` builtin with relaxed
 *     ordering (`__ATOMIC_RELAXED`): the read-modify-write is atomic but imposes
 *     no ordering on surrounding memory operations, which is all that is needed
 *     for the counter/accumulator use here.
 *
 * @tparam T Element type of the target; must be supported by the selected
 *           backend's atomic add (integer or supported floating-point).
 * @param address Pointer to the target location; must be non-null and, on the
 *                device, must address memory the calling thread may write.
 * @param value Amount added to `*address`.
 * @return The value stored at @p address immediately before the addition, so
 *         callers can use it as a unique reservation index (fetch-then-add).
 * @warning Relaxed ordering means the return value must not be relied on to
 *          synchronize access to any other memory.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
atomic_add(T* address, const T value) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    return __atomic_fetch_add(address, value, __ATOMIC_RELAXED);
#endif
}

}
