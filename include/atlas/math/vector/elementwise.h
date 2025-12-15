#pragma once
#include <atlas/math/detail/config.h>
#include <atlas/math/detail/ops.h>
#include <atlas/math/vector/expression.h>
#include <cmath>
#include <type_traits>

namespace atlas::math {
template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

bool
all(const VectorExpression<bool, E>& mask) noexcept {
    const E& m          = mask();
    const std::size_t n = m.size();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) if (!m[i]) return false;
    return true;
}

template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

bool
any(const VectorExpression<bool, E>& mask) noexcept {
    const E& m          = mask();
    const std::size_t n = m.size();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) if (m[i]) return true;
    return false;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
abs(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return VectorUnaryOperator<T, E, detail::Abs<T>>(expr());
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
sign(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return VectorUnaryOperator<T, E, detail::Sign<T>>(expr());
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
clamp(const E& x, expr_value_t<E> lo, expr_value_t<E> hi) noexcept {
    using T = expr_value_t<E>;
    return VectorUnaryOperator<T, E, detail::ClampScalar<T>>(x(), detail::ClampScalar<T>{ lo, hi });
}

template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
cmin(const EL& left, const ER& right) noexcept {
    using T = std::common_type_t<expr_value_t<EL>
                                 ,
                                 expr_value_t<ER>>;
    return VectorBinaryOperator<T, EL, ER, detail::CompareMin<T>>(left(), right());
}

template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
cmax(const EL& left, const ER& right) noexcept {
    using T = std::common_type_t<expr_value_t<EL>
                                 ,
                                 expr_value_t<ER>>;
    return VectorBinaryOperator<T, EL, ER, detail::CompareMax<T>>(left(), right());
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
saturate(const E& x) noexcept {
    using T = expr_value_t<E>;
    return clamp(x, T(0), T(1));
}

template <typename EM, VectorExpressionType ET, VectorExpressionType EF>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
select(const EM& mask, const ET& t, const EF& f) noexcept {
    using T = expr_value_t<ET>;
    static_assert(std::is_same_v<expr_value_t<ET>,
                                 expr_value_t<EF>>,
                  "select: value types must match"
            )
        ;
    return VectorSelect<T, EM, ET, EF>(mask(), t(), f());
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
normalize(const E& expr) noexcept {
    using T      = expr_value_t<E>;
    const T len2 = length_squared(expr);
    if (len2 == T(0)) return expr();
    const T inv = T(1) / static_cast<T>(std::sqrt(len2));
    return expr * inv;
}

template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
normalize_safe(const E& expr, const E& fallback) noexcept {
    using T      = expr_value_t<E>;
    const T len2 = length_squared(expr);
    if (len2 == T(0)) return fallback();
    const T inv = T(1) / static_cast<T>(std::sqrt(len2));
    return expr * inv;
}

template <VectorExpressionType EU, VectorExpressionType EV>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
project(const EU& u, const EV& v) noexcept {
    using T = std::common_type_t<expr_value_t<EU>
                                 ,
                                 expr_value_t<EV>>;
    const T vv = dot(v, v);
    if (vv == T(0)) return v * T(0);
    return v * (dot(u, v) / vv);
}

template <VectorExpressionType EU, VectorExpressionType EV>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
reject(const EU& u, const EV& v) noexcept {
    return u - project(u, v);
}

template <VectorExpressionType EI, VectorExpressionType EN>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
reflect(const EI& i, const EN& n) noexcept {
    using T = std::common_type_t<expr_value_t<EI>
                                 ,
                                 expr_value_t<EN>>;
    return i - n * (T(2) * dot(i, n));
}

template <VectorExpressionType EI, VectorExpressionType EN>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE

auto
refract(const EI& i, const EN& n, expr_value_t<EI> eta) noexcept {
    using T = std::common_type_t<expr_value_t<EI>
                                 ,
                                 expr_value_t<EN>>;
    const T d = dot(i, n);
    const T k = T(1) - eta * eta * (T(1) - d * d);
    if (k < T(0)) return i - i;
    return i * eta - n * (eta * d + static_cast<T>(std::sqrt(k)));
}
}