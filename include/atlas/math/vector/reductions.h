#pragma once
#include <atlas/math/detail/config.h>
#include <atlas/math/vector/expression.h>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace atlas::math {
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
sum(const E& expr) noexcept {
    const auto& e         = expr();
    const std::size_t n   = e.size();
    expr_value_t<E> accum = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) accum += e[i];
    return accum;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
min(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    expr_value_t<E> m   = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) if (e[i] < m) m = e[i];
    return m;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
max(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    expr_value_t<E> m   = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) if (m < e[i]) m = e[i];
    return m;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
mean(const E& expr) noexcept {
    return sum(expr) / static_cast<expr_value_t<E>>(expr().size());
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
length_squared(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& e       = expr();
    const std::size_t n = e.size();
    T accum             = T(0);
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) accum += e[i] * e[i];
    return accum;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

expr_value_t<E>
length(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(length_squared(expr)));
}

template <VectorExpressionType EA, VectorExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
dot(const EA& a, const EB& b) noexcept {
    using T = std::common_type_t<expr_value_t<EA>
                                 ,
                                 expr_value_t<EB>>;
    const auto& x       = a();
    const auto& y       = b();
    const std::size_t n = x.size();
    T accum             = T(0);
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) accum += x[i] * y[i];
    return accum;
}

template <VectorExpressionType EA, VectorExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
distance(const EA& a, const EB& b) noexcept {
    using T = std::common_type_t<expr_value_t<EA>
                                 ,
                                 expr_value_t<EB>>;
    return static_cast<T>(std::sqrt(length_squared(a - b)));
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

std::size_t
argmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i)
        if (e[i] < m) {
            m   = e[i];
            idx = i;
        }
    return idx;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

std::size_t
argmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i)
        if (m < e[i]) {
            m   = e[i];
            idx = i;
        }
    return idx;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

std::size_t
argabsmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = std::abs(e[0]);
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]);
        if (v < m) {
            m   = v;
            idx = i;
        }
    }
    return idx;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

std::size_t
argabsmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = std::abs(e[0]);
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]);
        if (m < v) {
            m   = v;
            idx = i;
        }
    }
    return idx;
}
}