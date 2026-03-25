#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <atlas/core/macros.h>
#include <iterator>
#include <thrust/iterator/zip_iterator.h>
#include <tuple>
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

#else

#include <iterator>
#include <tuple>

namespace atlas {

template <typename... Iterators>
class zip_iterator {
public:
    using tuple_type = std::tuple<Iterators...>;

    using value_type = std::tuple<typename std::iterator_traits<Iterators>::value_type...>;

    using reference = std::tuple<typename std::iterator_traits<Iterators>::reference...>;

    using iterator_category = std::forward_iterator_tag;

    zip_iterator() = default;

    explicit zip_iterator(tuple_type iterators)
        : iters_(iterators) { }

    reference
    operator*() const {
        return deref(std::index_sequence_for<Iterators...> {});
    }

    zip_iterator&
    operator++() {
        increment(std::index_sequence_for<Iterators...> {});
        return *this;
    }

    zip_iterator
    operator++(int) {
        zip_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool
    operator==(const zip_iterator& other) const {
        return iters_ == other.iters_;
    }

    bool
    operator!=(const zip_iterator& other) const {
        return !(*this == other);
    }

    zip_iterator
    operator+(std::ptrdiff_t n) const {
        return advance(n, std::index_sequence_for<Iterators...> {});
    }

    std::ptrdiff_t
    operator-(const zip_iterator& other) const {
        return distance(other, std::index_sequence_for<Iterators...> {});
    }

private:
    tuple_type iters_ {};

    template <std::size_t... I>
    reference
    deref(std::index_sequence<I...>) const {

        return reference(*std::get<I>(iters_)...);
    }

    template <std::size_t... I>
    void
    increment(std::index_sequence<I...>) {

        ((++std::get<I>(iters_)), ...);
    }

    template <std::size_t... I>
    zip_iterator
    advance(std::ptrdiff_t n, std::index_sequence<I...>) const {

        return zip_iterator(tuple_type(std::next(std::get<I>(iters_), n)...));
    }

    template <std::size_t... I>
    std::ptrdiff_t
    distance(const zip_iterator& other, std::index_sequence<I...>) const {
        (void)sizeof...(I);

        return std::distance(std::get<0>(other.iters_), std::get<0>(iters_));
    }
};

template <typename... Iterators>
inline zip_iterator<Iterators...>
make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<Iterators...>(t);
}

template <typename... Ts>
inline auto
make_zip_tuple(Ts&&... args) {
    return std::make_tuple(std::forward<Ts>(args)...);
}

template <std::size_t I, typename Tuple>
inline decltype(auto)
zip_get(Tuple&& t) {
    return std::get<I>(std::forward<Tuple>(t));
}

}
#endif
