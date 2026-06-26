#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <atlas/core/macros.h>
#include <thrust/tuple.h>

namespace atlas {

template <typename... Ts>
using tuple = thrust::tuple<Ts...>;

template <typename... Ts>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    tuple<Ts...>
    make_tuple(Ts... args) {
    return thrust::make_tuple(args...);
}

template <std::size_t I, typename Tuple>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE decltype(auto)
get(const Tuple& t) {
    return thrust::get<I>(t);
}

}

#else

#include <tuple>

namespace atlas {

template <typename... Ts>
using tuple = std::tuple<Ts...>;

template <typename... Ts>
inline tuple<Ts...>
make_tuple(Ts... args) {
    return std::make_tuple(args...);
}

template <std::size_t I, typename Tuple>
inline decltype(auto)
get(const Tuple& t) {
    return std::get<I>(t);
}

}
#endif