#pragma once
#include <atlas/math/detail/ops.h>
#include <atlas/math/matrix/matrix_expression.h>
#include <type_traits>
namespace atlas::math {

template <typename T, typename E>
using MatrixNeg = MatrixUnaryOperator<T, E, detail::Negate<T>>;

template <typename To, typename FromExpr>
using MatrixTypeCast = MatrixUnaryOperator<To, FromExpr, detail::TypeCast<expr_value_t<FromExpr>, To>>;

template <typename T, typename EL, typename ER>
using MatrixAdd = MatrixBinaryOperator<T, EL, ER, detail::Add<T>>;

template <typename T, typename EL, typename ER>
using MatrixSub = MatrixBinaryOperator<T, EL, ER, detail::Sub<T>>;

template <typename T, typename EL, typename ER>
using MatrixMul = MatrixBinaryOperator<T, EL, ER, detail::Mul<T>>;

template <typename T, typename EL, typename ER>
using MatrixDiv = MatrixBinaryOperator<T, EL, ER, detail::Div<T>>;

template <typename T, typename E>
using MatrixAddScalarR = MatrixScalarRight<T, E, detail::Add<T>>;

template <typename T, typename E>
using MatrixSubScalarR = MatrixScalarRight<T, E, detail::Sub<T>>;

template <typename T, typename E>
using MatrixMulScalarR = MatrixScalarRight<T, E, detail::Mul<T>>;

template <typename T, typename E>
using MatrixDivScalarR = MatrixScalarRight<T, E, detail::Div<T>>;

template <typename T, typename E>
using MatrixAddScalarL = MatrixScalarLeft<T, E, detail::Add<T>>;

template <typename T, typename E>
using MatrixSubScalarL = MatrixScalarLeft<T, E, detail::Sub<T>>;

template <typename T, typename E>
using MatrixMulScalarL = MatrixScalarLeft<T, E, detail::Mul<T>>;

template <typename T, typename E>
using MatrixDivScalarL = MatrixScalarLeft<T, E, detail::Div<T>>;

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e) noexcept {
    using T = expr_value_t<E>;

    return MatrixNeg<T, E>(e());
}

template <typename To, MatrixExpressionType From>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cast_to(const From& e) noexcept {

    return MatrixTypeCast<To, From>(e());
}

template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;

    return MatrixAdd<T, EL, ER>(l(), r());
}

template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixSub<T, EL, ER>(l(), r());
}

template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixMul<T, EL, ER>(l(), r());
}

template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixDiv<T, EL, ER>(l(), r());
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;

    return MatrixAddScalarR<T, E>(e(), s);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixSubScalarR<T, E>(e(), s);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixMulScalarR<T, E>(e(), s);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixDivScalarR<T, E>(e(), s);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;

    return MatrixAddScalarL<T, E>(s, e());
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixSubScalarL<T, E>(s, e());
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixMulScalarL<T, E>(s, e());
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixDivScalarL<T, E>(s, e());
}
}