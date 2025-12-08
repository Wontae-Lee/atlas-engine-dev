#pragma once
#include <type_traits>

namespace atlas {
#ifdef ATLAS_TASKING_CUDA
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
template <typename T>
using device_ptr = thrust::device_ptr<T>;
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}
template <typename T>
__host__ __device__ constexpr T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}
template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}
#else
template <typename T>
using device_ptr = T*;

template <typename T>
constexpr T*
raw_pointer_cast(T* p) noexcept {
    return p;
}

template <typename T>
constexpr const T*
raw_pointer_cast(const T* p) noexcept {
    return p;
}

template <typename T>
constexpr T*
raw_pointer_cast(device_ptr<T> p) noexcept {
    return p;
}

template <typename T>
constexpr const T*
raw_pointer_cast(device_ptr<const T> p) noexcept {
    return p;
}
#endif
}