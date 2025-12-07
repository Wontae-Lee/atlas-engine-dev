#ifndef INCLUDE_ATLAS_MATH_MATRIX_ELEMENTWISE_H
#define INCLUDE_ATLAS_MATH_MATRIX_ELEMENTWISE_H
#include <atlas/math/detail/config.h>
#include <atlas/math/detail/ops.h>
#include <atlas/math/matrix/expression.h>
#include <cmath>
#include <type_traits>
namespace atlas::math {
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
abs(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return MatrixUnaryOperator<T, E, detail::Abs<T>>(expr());
}
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
sign(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return MatrixUnaryOperator<T, E, detail::Sign<T>>(expr());
}
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
clamp(const E& x, expr_value_t<E> lo, expr_value_t<E> hi) noexcept {
    using T = expr_value_t<E>;
    return MatrixUnaryOperator<T, E, detail::ClampScalar<T>>(x(), detail::ClampScalar<T> { lo, hi });
}
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmin(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixBinaryOperator<T, EL, ER, detail::CompareMin<T>>(l(), r());
}
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmax(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixBinaryOperator<T, EL, ER, detail::CompareMax<T>>(l(), r());
}
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
saturate(const E& x) noexcept {
    using T = expr_value_t<E>;
    return clamp(x, T(0), T(1));
}
template <typename EM, MatrixExpressionType ET, MatrixExpressionType EF>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
select(const EM& mask, const ET& t, const EF& f) noexcept {
    using T = expr_value_t<ET>;
    static_assert(std::is_same_v<expr_value_t<ET>, expr_value_t<EF>>, "select: value types must match");
    return MatrixSelect<T, EM, ET, EF>(mask(), t(), f());
}
}
#endif