#pragma once
#include <atlas/math/detail/ops.h>
#include <atlas/math/matrix/matrix_expression.h>
#include <type_traits>
namespace atlas::math {
/**
 * @file matrix_operators.h
 * @brief Operator overloads and convenience aliases for matrix expression templates.
 *
 * @details
 * This header provides:
 * - Shorthand type aliases (e.g., `MatrixAdd`, `MatrixNeg`) that bind generic expression node
 *   templates (`MatrixUnaryOperator`, `MatrixBinaryOperator`, `MatrixScalarRight`,
 *   `MatrixScalarLeft`) to specific operator functors in `atlas::math::detail`.
 * - Free-function operator overloads for natural math syntax on matrix expressions.
 * - A `cast_to<To>(expr)` helper for lane-wise element type conversion.
 *
 * All operators build **lazy expression nodes** and do not materialize a concrete matrix.
 * The resulting expression trees are intended to be fused and evaluated efficiently by
 * the backend (inlined loops on CPU or fused kernels on GPU).
 *
 * @note
 * - Expression operands are stored by const reference inside nodes; ensure operand lifetimes
 *   exceed the lifetime of the returned expression.
 * - Scalar operands are stored by value (safe to pass temporaries).
 * - Shape compatibility (rows/cols) is assumed for binary operators; no runtime checks are performed.
 *
 * @see MatrixExpression, MatrixUnaryOperator, MatrixBinaryOperator,
 *      MatrixScalarRight, MatrixScalarLeft
 */
// ------------------------------------------------------------
// Alias helpers for common expression nodes
// ------------------------------------------------------------
/** @brief Unary negation node: computes `-e` element-wise. */
template <typename T, typename E>
using MatrixNeg = MatrixUnaryOperator<T, E, detail::Negate<T>>;
/**
 * @brief Lane-wise type cast node: casts each element from `From` to `To`.
 *
 * @tparam To Target scalar element type.
 * @tparam FromExpr Source expression type.
 *
 * @note Uses `detail::TypeCast<From, To>` where `From = expr_value_t<FromExpr>`.
 */
template <typename To, typename FromExpr>
using MatrixTypeCast = MatrixUnaryOperator<To, FromExpr, detail::TypeCast<expr_value_t<FromExpr>, To>>;
/** @brief Element-wise addition node: `l + r`. */
template <typename T, typename EL, typename ER>
using MatrixAdd = MatrixBinaryOperator<T, EL, ER, detail::Add<T>>;
/** @brief Element-wise subtraction node: `l - r`. */
template <typename T, typename EL, typename ER>
using MatrixSub = MatrixBinaryOperator<T, EL, ER, detail::Sub<T>>;
/** @brief Element-wise multiplication node: `l * r`. */
template <typename T, typename EL, typename ER>
using MatrixMul = MatrixBinaryOperator<T, EL, ER, detail::Mul<T>>;
/** @brief Element-wise division node: `l / r`. */
template <typename T, typename EL, typename ER>
using MatrixDiv = MatrixBinaryOperator<T, EL, ER, detail::Div<T>>;
/** @brief Expression-scalar (right) addition: `e + s`. */
template <typename T, typename E>
using MatrixAddScalarR = MatrixScalarRight<T, E, detail::Add<T>>;
/** @brief Expression-scalar (right) subtraction: `e - s`. */
template <typename T, typename E>
using MatrixSubScalarR = MatrixScalarRight<T, E, detail::Sub<T>>;
/** @brief Expression-scalar (right) multiplication: `e * s`. */
template <typename T, typename E>
using MatrixMulScalarR = MatrixScalarRight<T, E, detail::Mul<T>>;
/** @brief Expression-scalar (right) division: `e / s`. */
template <typename T, typename E>
using MatrixDivScalarR = MatrixScalarRight<T, E, detail::Div<T>>;
/** @brief Scalar-expression (left) addition: `s + e`. */
template <typename T, typename E>
using MatrixAddScalarL = MatrixScalarLeft<T, E, detail::Add<T>>;
/** @brief Scalar-expression (left) subtraction: `s - e`. */
template <typename T, typename E>
using MatrixSubScalarL = MatrixScalarLeft<T, E, detail::Sub<T>>;
/** @brief Scalar-expression (left) multiplication: `s * e`. */
template <typename T, typename E>
using MatrixMulScalarL = MatrixScalarLeft<T, E, detail::Mul<T>>;
/** @brief Scalar-expression (left) division: `s / e`. */
template <typename T, typename E>
using MatrixDivScalarL = MatrixScalarLeft<T, E, detail::Div<T>>;
// ------------------------------------------------------------
// Unary operators
// ------------------------------------------------------------
/**
 * @brief Builds an expression node for unary negation (`-e`).
 *
 * @tparam E Matrix expression type.
 * @param e Operand expression.
 * @return Lazy expression computing `-e(r,c)` for each element.
 *
 * @details
 * Constructs a `MatrixUnaryOperator` node with `detail::Negate<T>`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e) noexcept {
    using T = expr_value_t<E>;
    // Unwrap CRTP (`e()`) so the node stores a reference to the derived expression type.
    return MatrixNeg<T, E>(e());
}
/**
 * @brief Casts each element of a matrix expression to a different scalar type.
 *
 * @tparam To Target scalar element type.
 * @tparam From Source matrix expression type.
 * @param e Input expression.
 * @return Lazy expression computing `static_cast<To>(e(r,c))` per element.
 *
 * @details
 * This is a named helper (rather than an operator) to avoid ambiguous overload sets
 * and to make intent explicit at call sites.
 */
template <typename To, MatrixExpressionType From>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cast_to(const From& e) noexcept {
    // Build a unary cast node; operand stored by reference, functor stored by value.
    return MatrixTypeCast<To, From>(e());
}
// ------------------------------------------------------------
// Arithmetic operators (expression-expression)
// ------------------------------------------------------------
/**
 * @brief Builds an expression node for element-wise addition (`l + r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `l(r,c) + r(r,c)`.
 *
 * @details
 * Result scalar type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 *
 * @warning
 * Assumes `l` and `r` have identical shape; no runtime checks are performed.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Store references to derived nodes; build binary operator node.
    return MatrixAdd<T, EL, ER>(l(), r());
}
/**
 * @brief Builds an expression node for element-wise subtraction (`l - r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `l(r,c) - r(r,c)`.
 *
 * @details
 * Result scalar type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 *
 * @warning
 * Assumes `l` and `r` have identical shape; no runtime checks are performed.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixSub<T, EL, ER>(l(), r());
}
/**
 * @brief Builds an expression node for element-wise multiplication (`l * r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `l(r,c) * r(r,c)`.
 *
 * @details
 * Result scalar type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 *
 * @warning
 * Assumes `l` and `r` have identical shape; no runtime checks are performed.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixMul<T, EL, ER>(l(), r());
}
/**
 * @brief Builds an expression node for element-wise division (`l / r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `l(r,c) / r(r,c)`.
 *
 * @details
 * Result scalar type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 *
 * @warning
 * Assumes `l` and `r` have identical shape; no runtime checks are performed.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return MatrixDiv<T, EL, ER>(l(), r());
}
// ------------------------------------------------------------
// Arithmetic operators (expression-scalar): e op s
// ------------------------------------------------------------
/**
 * @brief Builds an expression node for adding a scalar to a matrix expression (`e + s`).
 *
 * @tparam E Matrix expression type.
 * @param e Matrix expression operand.
 * @param s Scalar value (right operand).
 * @return Lazy expression computing `e(r,c) + s`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    // Scalar captured by value inside MatrixScalarRight.
    return MatrixAddScalarR<T, E>(e(), s);
}
/**
 * @brief Builds an expression node for subtracting a scalar from a matrix expression (`e - s`).
 *
 * @tparam E Matrix expression type.
 * @param e Matrix expression operand.
 * @param s Scalar value (right operand).
 * @return Lazy expression computing `e(r,c) - s`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixSubScalarR<T, E>(e(), s);
}
/**
 * @brief Builds an expression node for multiplying a matrix expression by a scalar (`e * s`).
 *
 * @tparam E Matrix expression type.
 * @param e Matrix expression operand.
 * @param s Scalar value (right operand).
 * @return Lazy expression computing `e(r,c) * s`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixMulScalarR<T, E>(e(), s);
}
/**
 * @brief Builds an expression node for dividing a matrix expression by a scalar (`e / s`).
 *
 * @tparam E Matrix expression type.
 * @param e Matrix expression operand.
 * @param s Scalar value (right operand).
 * @return Lazy expression computing `e(r,c) / s`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const E& e, expr_value_t<E> s) noexcept {
    using T = expr_value_t<E>;
    return MatrixDivScalarR<T, E>(e(), s);
}
// ------------------------------------------------------------
// Arithmetic operators (scalar-expression): s op e
// ------------------------------------------------------------
/**
 * @brief Builds an expression node for adding a matrix expression to a scalar (`s + e`).
 *
 * @tparam E Matrix expression type.
 * @param s Scalar value (left operand).
 * @param e Matrix expression operand.
 * @return Lazy expression computing `s + e(r,c)`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    // Scalar captured by value; expression stored by reference (unwrapped).
    return MatrixAddScalarL<T, E>(s, e());
}
/**
 * @brief Builds an expression node for subtracting a matrix expression from a scalar (`s - e`).
 *
 * @tparam E Matrix expression type.
 * @param s Scalar value (left operand).
 * @param e Matrix expression operand.
 * @return Lazy expression computing `s - e(r,c)`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixSubScalarL<T, E>(s, e());
}
/**
 * @brief Builds an expression node for multiplying a matrix expression by a scalar (`s * e`).
 *
 * @tparam E Matrix expression type.
 * @param s Scalar value (left operand).
 * @param e Matrix expression operand.
 * @return Lazy expression computing `s * e(r,c)`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixMulScalarL<T, E>(s, e());
}
/**
 * @brief Builds an expression node for dividing a scalar by a matrix expression (`s / e`).
 *
 * @tparam E Matrix expression type.
 * @param s Scalar value (left operand).
 * @param e Matrix expression operand.
 * @return Lazy expression computing `s / e(r,c)`.
 *
 * @note
 * This is element-wise scalar division by matrix elements, not a linear-algebra solve.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(expr_value_t<E> s, const E& e) noexcept {
    using T = expr_value_t<E>;
    return MatrixDivScalarL<T, E>(s, e());
}
} // namespace atlas::math
