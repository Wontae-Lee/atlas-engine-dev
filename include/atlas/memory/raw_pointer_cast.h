#ifndef INCLUDE_ATLAS_MEMORY_RAW_POINTER_CAST_H
#define INCLUDE_ATLAS_MEMORY_RAW_POINTER_CAST_H

#include <type_traits>

namespace atlas {

#ifdef ATLAS_TASKING_CUDA
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>


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
raw_pointer_cast(thrust::device_ptr<T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

template <typename T>
__host__ __device__ constexpr const T*
raw_pointer_cast(thrust::device_ptr<const T> p) noexcept {
    return thrust::raw_pointer_cast(p);
}

#else

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

#endif

}
#endif