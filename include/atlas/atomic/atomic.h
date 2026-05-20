#pragma once

#include <atlas/core/macros.h>

#if defined(ATLAS_TASKING_CUDA)
#include <cuda_runtime.h>
#endif

#if !defined(ATLAS_TASKING_CUDA) || !defined(__CUDA_ARCH__)
#include <atomic>
#endif

namespace atlas {

/**
 * @brief Atomically add an integer with relaxed ordering.
 *
 * @return The value stored at @p address before the addition.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
atomic_fetch_add_relaxed(int* address, const int value) noexcept {
#if defined(ATLAS_TASKING_CUDA) && defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    std::atomic_ref<int> atomic_value(*address);
    return atomic_value.fetch_add(value, std::memory_order_relaxed);
#endif
}

/**
 * @brief Atomically compare and exchange an integer with acquire semantics.
 *
 * @return The value stored at @p address before the compare/exchange attempt.
 */
ATLAS_DEVICE ATLAS_FORCE_INLINE int
atomic_compare_exchange_acquire(int* address, const int expected, const int desired) noexcept {
#if defined(ATLAS_TASKING_CUDA)
    return atomicCAS(address, expected, desired);
#else
    std::atomic_ref<int> atomic_value(*address);
    int expected_value = expected;

    while (!atomic_value.compare_exchange_weak(
        expected_value,
        desired,
        std::memory_order_acquire,
        std::memory_order_relaxed)) {
        if (expected_value != expected) {
            return expected_value;
        }
    }

    return expected;
#endif
}

/**
 * @brief Atomically exchange an integer with release semantics.
 *
 * @return The value stored at @p address before the exchange.
 */
ATLAS_DEVICE ATLAS_FORCE_INLINE int
atomic_exchange_release(int* address, const int desired) noexcept {
#if defined(ATLAS_TASKING_CUDA)
    return atomicExch(address, desired);
#else
    std::atomic_ref<int> atomic_value(*address);
    return atomic_value.exchange(desired, std::memory_order_release);
#endif
}

} // namespace atlas
