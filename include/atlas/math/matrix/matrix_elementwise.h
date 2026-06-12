#pragma once
#include <atlas/math/detail/ops.h>
#include <atlas/math/matrix/matrix_expression.h>
#include <cmath>
#include <type_traits>
namespace atlas {
/**
 * @file matrix_elementwise.h
 * @brief Element-wise utility operations for matrix expression templates.
 *
 * @details
 * This header defines small helper functions for Atlas matrix expression templates.
 * Most functions return **lazy expression nodes** (e.g., `MatrixUnaryOperator`,
 * `MatrixBinaryOperator`, `MatrixSelect`) rather than materializing a concrete matrix.
 *
 * These utilities are intended to be composable and fuse-friendly:
 * chaining multiple operations should produce a single expression tree that the backend
 * can evaluate efficiently (e.g., inlined loops or fused GPU kernels).
 *
 * Provided operations:
 * - Unary: `abs`, `sign`
 * - Unary-with-scalar-params: `clamp`, `saturate`
 * - Binary: `cmin`, `cmax` (element-wise min/max)
 * - Ternary: `select` (mask-based element-wise choose)
 *
 * @note
 * - Expression nodes generally store operands by const reference; ensure operand lifetimes
 *   exceed the lifetime of the returned expression.
 * - Scalar parameters (e.g., clamp bounds) are captured inside the operator functor by value.
 *
 * @see atlas::MatrixExpression, atlas::MatrixUnaryOperator,
 *      atlas::MatrixBinaryOperator, atlas::MatrixSelect
 */
// ------------------------------------------------------------
// Unary element-wise ops
// ------------------------------------------------------------
/**
 * @brief Element-wise absolute value for a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input matrix expression.
 * @return Lazy expression computing `abs(expr(r,c))` for each element.
 *
 * @details
 * Builds a `MatrixUnaryOperator` node using `detail::Abs<T>`.
 * No computation is performed until the expression is evaluated.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
abs(const E& expr) noexcept {
    using T = expr_value_t<E>; // Scalar element type.
    // Build a unary expression node: y = Abs(x) element-wise.
    return MatrixUnaryOperator<T, E, detail::Abs<T>>(expr());
}
/**
 * @brief Element-wise sign function for a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input matrix expression.
 * @return Lazy expression computing `sign(expr(r,c))` for each element.
 *
 * @details
 * Builds a `MatrixUnaryOperator` node using `detail::Sign<T>`.
 * Typical convention:
 * - negative -> -1
 * - zero     ->  0
 * - positive -> +1
 *
 * @note
 * Exact NaN handling (if applicable) depends on `detail::Sign<T>`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
sign(const E& expr) noexcept {
    using T = expr_value_t<E>;
    // Build a unary expression node: y = Sign(x) element-wise.
    return MatrixUnaryOperator<T, E, detail::Sign<T>>(expr());
}
// ------------------------------------------------------------
// Clamp / saturate
// ------------------------------------------------------------
/**
 * @brief Element-wise clamp with scalar bounds for a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param x  Input expression.
 * @param lo Lower scalar bound.
 * @param hi Upper scalar bound.
 * @return Lazy expression computing `clamp(x(r,c), lo, hi)` for each element.
 *
 * @details
 * Uses `detail::ClampScalar<T>` as the per-element operator. Bounds are stored inside
 * the functor (by value), so passing temporaries is safe.
 *
 * @note
 * No explicit validation of `lo <= hi` is performed; behavior is determined by `ClampScalar`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
clamp(const E& x, expr_value_t<E> lo, expr_value_t<E> hi) noexcept {
    using T = expr_value_t<E>;
    // The clamp functor captures the bounds.
    return MatrixUnaryOperator<T, E, detail::ClampScalar<T>>(x(),
                                                             detail::ClampScalar<T> { lo, hi });
}
/**
 * @brief Element-wise saturate: clamp values into [0, 1].
 *
 * @tparam E Matrix expression type.
 * @param x Input expression.
 * @return Lazy expression computing `clamp(x(r,c), 0, 1)` for each element.
 *
 * @details
 * This is a convenience wrapper over `clamp(...)` using `T(0)` and `T(1)` as bounds.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
saturate(const E& x) noexcept {
    using T = expr_value_t<E>;
    // Reuse clamp with canonical bounds.
    return clamp(x, T(0), T(1));
}
// ------------------------------------------------------------
// Element-wise compare min/max
// ------------------------------------------------------------
/**
 * @brief Element-wise minimum between two matrix expressions.
 *
 * @tparam EL Left matrix expression type.
 * @tparam ER Right matrix expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `min(l(r,c), r(r,c))` element-wise.
 *
 * @details
 * Result element type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 * Implemented via `detail::CompareMin<T>`.
 *
 * @warning
 * Assumes both matrices have identical shape (rows/cols). Shape mismatches are undefined.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmin(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Build a binary operator node: z = CompareMin(x, y) element-wise.
    return MatrixBinaryOperator<T, EL, ER, detail::CompareMin<T>>(l(), r());
}
/**
 * @brief Element-wise maximum between two matrix expressions.
 *
 * @tparam EL Left matrix expression type.
 * @tparam ER Right matrix expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression computing `max(l(r,c), r(r,c))` element-wise.
 *
 * @details
 * Result element type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 * Implemented via `detail::CompareMax<T>`.
 *
 * @warning
 * Assumes both matrices have identical shape (rows/cols). Shape mismatches are undefined.
 */
template <MatrixExpressionType EL, MatrixExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmax(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Build a binary operator node: z = CompareMax(x, y) element-wise.
    return MatrixBinaryOperator<T, EL, ER, detail::CompareMax<T>>(l(), r());
}
// ------------------------------------------------------------
// Select
// ------------------------------------------------------------
/**
 * @brief Element-wise select between two matrix expressions using a mask.
 *
 * @tparam EM Mask expression type (expects `mask(r,c)` or `mask[i]` semantics via `mask()`).
 * @tparam ET True-branch matrix expression type.
 * @tparam EF False-branch matrix expression type.
 * @param mask Mask expression; if mask element is true, select from `t`, else from `f`.
 * @param t True-branch expression.
 * @param f False-branch expression.
 * @return Lazy expression selecting element-wise between `t` and `f`.
 *
 * @details
 * Enforces that value types of `t` and `f` match exactly via `static_assert`.
 * The returned node is `MatrixSelect<T, EM, ET, EF>`.
 *
 * @note
 * The mask type is not constrained to `MatrixExpressionType` to allow flexible mask wrappers,
 * but it must provide compatible element access when evaluated by `MatrixSelect`.
 */
template <typename EM, MatrixExpressionType ET, MatrixExpressionType EF>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
select(const EM& mask, const ET& t, const EF& f) noexcept {
    using T = expr_value_t<ET>;
    static_assert(std::is_same_v<expr_value_t<ET>, expr_value_t<EF>>,
                  "select: value types must match");
    // Build a select expression node. The node stores references to mask/t/f.
    return MatrixSelect<T, EM, ET, EF>(mask(), t(), f());
}
} // namespace atlas
