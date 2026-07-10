#pragma once

#include <atlas/core/macros.h>

namespace atlas {

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
