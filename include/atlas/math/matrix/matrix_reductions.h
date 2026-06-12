#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
namespace atlas {

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
sum(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    if (n == 0) return expr_value_t<E>(0);
    expr_value_t<E> acc = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        acc += e[i];
    }
    return acc;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
min(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (e[i] < m) m = e[i];
    }
    return m;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
max(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (m < e[i]) m = e[i];
    }
    return m;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
length_squared(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& e       = expr();
    const std::size_t n = e.size();
    T acc               = T(0);
    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            const T v = e[i];
            acc       = fma(v, v, acc);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc += e[i] * e[i];
        }
    }
    return acc;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
length(const E& expr) noexcept {
    using T = expr_value_t<E>;
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared(expr)));
}

template <MatrixExpressionType EA, MatrixExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
dot(const EA& a, const EB& b) noexcept {
    using T             = std::common_type_t<expr_value_t<EA>, expr_value_t<EB>>;
    const auto& x       = a();
    const auto& y       = b();
    const std::size_t n = x.size();
    T acc               = T(0);
    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc = fma(static_cast<T>(x[i]), static_cast<T>(y[i]), acc);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc += x[i] * y[i];
        }
    }
    return acc;
}

template <MatrixExpressionType EA, MatrixExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
distance(const EA& a, const EB& b) noexcept {
    using T         = std::common_type_t<expr_value_t<EA>, expr_value_t<EB>>;
    const auto diff = a - b;
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared(diff)));
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (e[i] < m) {
            m   = e[i];
            idx = i;
        }
    }
    return idx;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (m < e[i]) {
            m   = e[i];
            idx = i;
        }
    }
    return idx;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argabsmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto mm             = std::abs(e[0]);
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]);
        if (v < mm) {
            mm  = v;
            idx = i;
        }
    }
    return idx;
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argabsmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto mm             = std::abs(e[0]);
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]);
        if (mm < v) {
            mm  = v;
            idx = i;
        }
    }
    return idx;
}

}