#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <cmath>
#include <cstddef>
#include <thrust/device_reference.h>
#include <type_traits>

namespace atlas::test {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
static ATLAS_FORCE_INLINE T
unwrap(const thrust::device_reference<T>& ref) {
    return ref;
}

template <typename T>
static ATLAS_FORCE_INLINE T
unwrap(const thrust::device_reference<const T>& ref) {
    return ref;
}

template <typename T>
static ATLAS_FORCE_INLINE const T&
unwrap(const T& value) {
    return value;
}

template <typename V>
static ATLAS_FORCE_INLINE std::size_t
vec_size(const V& v) {
    if constexpr (requires { remove_cvref_t<V>::size(); }) {
        return remove_cvref_t<V>::size();
    } else {
        return v.size();
    }
}

template <typename T>
static ATLAS_FORCE_INLINE bool
near(T a, T b, T eps) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= eps;
    } else {
        return a == b;
    }
}

template <typename V>
static ATLAS_FORCE_INLINE bool
is_finite_vec(const V& v) {
    if constexpr (std::is_convertible_v<decltype(v[0]), double>) {
        for (std::size_t i = 0; i < vec_size(v); ++i) {
            if (!std::isfinite(static_cast<double>(v[i]))) return false;
        }
    }

    return true;
}

namespace detail {

    template <typename A, typename B, typename Eps>
    static ATLAS_FORCE_INLINE bool
    vec_near_impl(const A& a,
                  const B& b,
                  Eps eps) {
        if constexpr (!std::is_same_v<remove_cvref_t<A>, remove_cvref_t<B>>
                      && (std::is_convertible_v<A, remove_cvref_t<B>>
                          || std::is_constructible_v<remove_cvref_t<B>, A>)) {
            return vec_near_impl(static_cast<remove_cvref_t<B>>(a), b, eps);
        } else if constexpr (!std::is_same_v<remove_cvref_t<A>, remove_cvref_t<B>>
                             && (std::is_convertible_v<B, remove_cvref_t<A>>
                                 || std::is_constructible_v<remove_cvref_t<A>, B>)) {
            return vec_near_impl(a, static_cast<remove_cvref_t<A>>(b), eps);
        }

        const std::size_t n = vec_size(b);

        for (std::size_t i = 0; i < n; ++i) {
            using T = std::common_type_t<remove_cvref_t<decltype(a[i])>,
                                         remove_cvref_t<decltype(b[i])>,
                                         remove_cvref_t<Eps>>;

            if (!near<T>(static_cast<T>(a[i]),
                         static_cast<T>(b[i]),
                         static_cast<T>(eps))) {
                return false;
            }
        }

        return true;
    }

}

template <typename A, typename B, typename Eps>
static ATLAS_FORCE_INLINE bool
vec_near(const A& a,
         const B& b,
         Eps eps) {
    return detail::vec_near_impl(unwrap(a), unwrap(b), eps);
}

} // namespace atlas::test
