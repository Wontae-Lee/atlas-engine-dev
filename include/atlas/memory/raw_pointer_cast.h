#pragma once

#include <atlas/core/macros.h>

#include <thrust/device_ptr.h>

namespace atlas {

template <typename T>
using device_ptr = thrust::device_ptr<T>;

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

}
