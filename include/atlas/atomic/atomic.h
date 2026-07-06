#pragma once

#include <atlas/core/macros.h>

#if defined(ATLAS_TASKING_CUDA)
#include <cuda_runtime.h>
#endif

#if !defined(ATLAS_TASKING_CUDA) || !defined(__CUDA_ARCH__)
#include <atomic>
#endif

namespace atlas {

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
atomic_fetch_add_relaxed(int* address, const int value) noexcept {
#if defined(ATLAS_TASKING_CUDA) && defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    std::atomic_ref<int> atomic_value(*address);
    return atomic_value.fetch_add(value, std::memory_order_relaxed);
#endif
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
atomic_compare_exchange_acquire(int* address, const int expected, const int desired) noexcept {
#if defined(ATLAS_TASKING_CUDA) && defined(__CUDA_ARCH__)
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

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
atomic_exchange_release(int* address, const int desired) noexcept {
#if defined(ATLAS_TASKING_CUDA) && defined(__CUDA_ARCH__)
    return atomicExch(address, desired);
#else
    std::atomic_ref<int> atomic_value(*address);
    return atomic_value.exchange(desired, std::memory_order_release);
#endif
}

}