#pragma once

#include <atlas/core/macros.h>
#include <cstddef>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <utility>

namespace atlas {

template <typename... Iterators>
using zip_iterator = thrust::zip_iterator<thrust::tuple<Iterators...>>;

template <typename... Iterators>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE zip_iterator<Iterators...>
make_zip_iterator(thrust::tuple<Iterators...> t) {
    return thrust::make_zip_iterator(t);
}

template <typename... Ts>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
make_zip_tuple(Ts&&... args) {
    return thrust::make_tuple(std::forward<Ts>(args)...);
}

template <std::size_t I, typename Tuple>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE decltype(auto)
zip_get(Tuple&& t) {
    return thrust::get<I>(std::forward<Tuple>(t));
}

}
