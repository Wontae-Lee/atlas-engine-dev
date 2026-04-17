#pragma once

#include <atlas/math/detail/ops.h>
#include <atlas/math/vector/vector_expression.h>
#include <cmath>
#include <type_traits>

namespace atlas::math {

/**
 * @file vector_elementwise.h
 * @brief Vector expression utilities (predicates and common math operators).
 *
 * @details
 * This header defines small, composable helpers operating on Atlas vector expression templates.
 * Most functions return expression nodes (lazy evaluation) rather than materialized vectors,
 * allowing the compiler/back-end to fuse multiple operations into a single kernel/loop.
 *
 * The only exceptions are boolean reductions such as `all()` / `any()`, which must iterate the
 * underlying mask expression and return a scalar `bool`.
 *
 * @note
 * - Functions are marked `ATLAS_ALL_DEVICE` and can be called from both host and device code.
 * - Unary/binary operators return expression objects and do not allocate memory.
 * - Reductions (`all`, `any`) may cause immediate evaluation of the mask expression.
 */

/**
 * @brief Returns true if all lanes of a boolean mask evaluate to true.
 *
 * @tparam E Underlying mask expression type.
 * @param mask Boolean vector expression.
 * @return `true` if every element is `true`; otherwise `false`.
 *
 * @details
 * This is a short-circuiting reduction:
 * it returns immediately on the first `false` element.
 *
 * @note
 * - The mask is evaluated element-wise via `operator[]`.
 * - Uses `ATLAS_UNROLL` to encourage unrolling for small fixed-size vectors.
 */
template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const VectorExpression<bool, E>& mask) noexcept {
    const E& m          = mask();   // Materialize the expression reference (no copy of data).
    const std::size_t n = m.size(); // Runtime size; for fixed-size vectors this is constant-folded.

    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Early-out as soon as a lane is false.
        if (!m[i]) return false;
    }
    return true; // All lanes were true.
}

/**
 * @brief Returns true if any lane of a boolean mask evaluates to true.
 *
 * @tparam E Underlying mask expression type.
 * @param mask Boolean vector expression.
 * @return `true` if at least one element is `true`; otherwise `false`.
 *
 * @details
 * This is a short-circuiting reduction:
 * it returns immediately on the first `true` element.
 *
 * @note
 * - The mask is evaluated element-wise via `operator[]`.
 * - Uses `ATLAS_UNROLL` to encourage unrolling for small fixed-size vectors.
 */
template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const VectorExpression<bool, E>& mask) noexcept {
    const E& m          = mask();   // Bind to the underlying expression node.
    const std::size_t n = m.size(); // Number of lanes.

    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Early-out as soon as a lane is true.
        if (m[i]) return true;
    }
    return false; // No lanes were true.
}

/**
 * @brief Element-wise absolute value.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Lazy expression computing `abs(expr[i])` for each lane.
 *
 * @details
 * Returns a unary expression node using `detail::Abs<T>`.
 * No computation is performed until the returned expression is evaluated.
 *
 * @note
 * The absolute operation is performed per lane; the exact semantics depend on `detail::Abs<T>`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
abs(const E& expr) noexcept {
    using T = expr_value_t<E>; // Scalar lane type.
    // Build a unary operator expression node: y = Abs(x).
    return VectorUnaryOperator<T, E, detail::Abs<T>>(expr());
}

/**
 * @brief Element-wise sign function.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Lazy expression computing `sign(expr[i])` for each lane.
 *
 * @details
 * Returns a unary expression node using `detail::Sign<T>`.
 * Typical convention is:
 * - negative -> -1
 * - zero     ->  0
 * - positive -> +1
 *
 * @note
 * The exact behavior for NaNs (if applicable) depends on `detail::Sign<T>`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
sign(const E& expr) noexcept {
    using T = expr_value_t<E>;
    // Build a unary operator expression node: y = Sign(x).
    return VectorUnaryOperator<T, E, detail::Sign<T>>(expr());
}

/**
 * @brief Element-wise clamp with scalar bounds.
 *
 * @tparam E Vector expression type.
 * @param x  Input expression.
 * @param lo Lower scalar bound.
 * @param hi Upper scalar bound.
 * @return Lazy expression computing `clamp(x[i], lo, hi)` per lane.
 *
 * @details
 * The clamping is performed by `detail::ClampScalar<T>` using scalar parameters `{lo, hi}`.
 *
 * @note
 * No check is performed to ensure `lo <= hi`; behavior is defined by `ClampScalar`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
clamp(const E& x, expr_value_t<E> lo, expr_value_t<E> hi) noexcept {
    using T = expr_value_t<E>;
    // Build a unary operator node that captures scalar bounds.
    return VectorUnaryOperator<T, E, detail::ClampScalar<T>>(x(),
                                                             detail::ClampScalar<T> { lo, hi });
}

/**
 * @brief Element-wise compare-min between two expressions.
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param left  Left input expression.
 * @param right Right input expression.
 * @return Lazy expression computing `min(left[i], right[i])` per lane.
 *
 * @details
 * Result lane type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 * The operation is implemented via `detail::CompareMin<T>`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmin(const EL& left, const ER& right) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Build a binary operator expression node: z = CompareMin(x, y).
    return VectorBinaryOperator<T, EL, ER, detail::CompareMin<T>>(left(), right());
}

/**
 * @brief Element-wise compare-max between two expressions.
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param left  Left input expression.
 * @param right Right input expression.
 * @return Lazy expression computing `max(left[i], right[i])` per lane.
 *
 * @details
 * Result lane type is `std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>`.
 * The operation is implemented via `detail::CompareMax<T>`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
cmax(const EL& left, const ER& right) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Build a binary operator expression node: z = CompareMax(x, y).
    return VectorBinaryOperator<T, EL, ER, detail::CompareMax<T>>(left(), right());
}

/**
 * @brief Element-wise saturate: clamp values into [0, 1].
 *
 * @tparam E Vector expression type.
 * @param x Input expression.
 * @return Lazy expression computing `clamp(x[i], 0, 1)` per lane.
 *
 * @note
 * Uses the value type of the expression for bounds, i.e. `T(0)` and `T(1)`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
saturate(const E& x) noexcept {
    using T = expr_value_t<E>;
    // Reuse clamp() with canonical [0, 1] bounds.
    return clamp(x, T(0), T(1));
}

/**
 * @brief Element-wise select between two expressions using a mask.
 *
 * @tparam EM Mask expression type (typically bool vector expression).
 * @tparam ET True-branch expression type.
 * @tparam EF False-branch expression type.
 * @param mask Mask expression; when mask[i] is true selects t[i], else f[i].
 * @param t    Expression providing values for true lanes.
 * @param f    Expression providing values for false lanes.
 * @return Lazy expression selecting per lane from `t` or `f`.
 *
 * @details
 * The value types of `t` and `f` must match exactly.
 * This is enforced by a `static_assert`.
 *
 * @note
 * `mask` is not constrained to `VectorExpressionType` here to allow flexible mask wrappers,
 * but it is expected to be a compatible mask expression in practice.
 */
template <typename EM, VectorExpressionType ET, VectorExpressionType EF>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
select(const EM& mask, const ET& t, const EF& f) noexcept {
    using T = expr_value_t<ET>;
    static_assert(std::is_same_v<expr_value_t<ET>, expr_value_t<EF>>,
                  "select: value types must match");
    // Build a select expression node that branches per lane.
    return VectorSelect<T, EM, ET, EF>(mask(), t(), f());
}

/**
 * @brief Normalizes a vector expression (returns a unit-length vector).
 *
 * @tparam E Vector expression type.
 * @param expr Input vector expression.
 * @return Lazy expression representing `expr / ||expr||`, or `expr` if `||expr|| == 0`.
 *
 * @details
 * Computes the squared length `len2 = length_squared(expr)` and returns:
 * - `expr()` if `len2 == 0` (to avoid division by zero)
 * - `expr * (1 / sqrt(len2))` otherwise
 *
 * @note
 * - The equality check `len2 == 0` is exact; for floating-point robustness prefer `normalize_safe`
 *   with an epsilon-based policy at a higher level if needed.
 * - Returning `expr()` preserves the original expression when normalization is undefined.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
normalize(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const T len2 = length_squared(expr); // Length squared; avoids an immediate sqrt when checking zero.
    if (len2 == T(0)) {
        // Degenerate vector: return the original expression to avoid NaNs/infs.
        return expr();
    }

    const T inv = T(1) / static_cast<T>(std::sqrt(len2)); // Compute reciprocal length.
    return expr * inv;                                    // Scale to unit length (lazy multiply expression).
}

/**
 * @brief Normalizes a vector expression; returns a fallback if the input is degenerate.
 *
 * @tparam E Vector expression type.
 * @param expr     Input vector expression.
 * @param fallback Expression returned when `||expr|| == 0`.
 * @return Normalized expression, or `fallback()` when degenerate.
 *
 * @details
 * This variant avoids undefined normalization by returning the provided fallback expression
 * when the input has zero length.
 *
 * @note
 * - The fallback expression is returned as-is (`fallback()`), preserving laziness.
 * - The equality check is exact (`len2 == 0`).
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
normalize_safe(const E& expr, const E& fallback) noexcept {
    using T = expr_value_t<E>;

    const T len2 = length_squared(expr); // Squared magnitude.
    if (len2 == T(0)) {
        // Degenerate vector: use caller-provided fallback.
        return fallback();
    }

    const T inv = T(1) / static_cast<T>(std::sqrt(len2)); // Reciprocal magnitude.
    return expr * inv;                                    // Scale to unit length.
}

/**
 * @brief Projects vector `u` onto vector `v`.
 *
 * @tparam EU Expression type for u.
 * @tparam EV Expression type for v.
 * @param u Vector to be projected.
 * @param v Projection direction (onto this vector).
 * @return Projection of u onto v: `v * (dot(u,v) / dot(v,v))`.
 *
 * @details
 * Uses the standard vector projection formula:
 * \f[
 *   \text{proj}_\mathbf{v}(\mathbf{u}) =
 *   \mathbf{v}\,\frac{\mathbf{u}\cdot\mathbf{v}}{\mathbf{v}\cdot\mathbf{v}}.
 * \f]
 *
 * If `v` is the zero vector (`dot(v,v) == 0`), returns `v * 0` (a zero vector expression)
 * to avoid division by zero.
 *
 * @note
 * - The result lane type is the common type of the input lane types.
 * - `v * T(0)` intentionally preserves the shape/dimension of `v`.
 */
template <VectorExpressionType EU, VectorExpressionType EV>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
project(const EU& u, const EV& v) noexcept {
    using T = std::common_type_t<expr_value_t<EU>, expr_value_t<EV>>;

    const T vv = dot(v, v); // Denominator: squared length of v.
    if (vv == T(0)) {
        // Degenerate direction: projection is defined as zero to avoid division by zero.
        return v * T(0);
    }

    // Scalar scale factor (u·v)/(v·v), applied to v.
    return v * (dot(u, v) / vv);
}

/**
 * @brief Rejects (removes) the component of `u` along `v`.
 *
 * @tparam EU Expression type for u.
 * @tparam EV Expression type for v.
 * @param u Input vector.
 * @param v Direction being removed.
 * @return Rejection of u from v: `u - project(u, v)`.
 *
 * @details
 * The rejection is the component orthogonal to `v`:
 * \f[
 *   \text{rej}_\mathbf{v}(\mathbf{u}) =
 *   \mathbf{u} - \text{proj}_\mathbf{v}(\mathbf{u}).
 * \f]
 */
template <VectorExpressionType EU, VectorExpressionType EV>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
reject(const EU& u, const EV& v) noexcept {
    // Compose projection and subtraction; both are expression nodes.
    return u - project(u, v);
}

/**
 * @brief Reflects incident vector `i` about normal vector `n`.
 *
 * @tparam EI Expression type for incident vector.
 * @tparam EN Expression type for normal vector.
 * @param i Incident direction.
 * @param n Surface normal (does not need to be unit length, but typical usage assumes normalized).
 * @return Reflected direction.
 *
 * @details
 * Uses the standard reflection formula:
 * \f[
 *   \mathbf{r} = \mathbf{i} - 2(\mathbf{i}\cdot\mathbf{n})\mathbf{n}.
 * \f]
 *
 * @note
 * If `n` is not normalized, the reflected magnitude will be scaled accordingly.
 */
template <VectorExpressionType EI, VectorExpressionType EN>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
reflect(const EI& i, const EN& n) noexcept {
    using T = std::common_type_t<expr_value_t<EI>, expr_value_t<EN>>;
    // r = i - 2 * dot(i, n) * n
    return i - n * (T(2) * dot(i, n));
}

/**
 * @brief Refracts incident vector `i` through a surface with normal `n`.
 *
 * @tparam EI Expression type for incident vector.
 * @tparam EN Expression type for normal vector.
 * @param i   Incident direction.
 * @param n   Surface normal (typically unit length).
 * @param eta Relative index of refraction (IOR ratio), usually `eta_i / eta_t`.
 * @return Refracted direction, or zero vector if total internal reflection occurs.
 *
 * @details
 * This implements a common refraction form consistent with Snell's law.
 * Let \f$d = \mathbf{i}\cdot\mathbf{n}\f$ and
 * \f[
 *   k = 1 - \eta^2 (1 - d^2).
 * \f]
 * If \f$k < 0\f$, total internal reflection occurs and the function returns a zero vector.
 * Otherwise:
 * \f[
 *   \mathbf{t} = \eta \mathbf{i} - \left(\eta d + \sqrt{k}\right)\mathbf{n}.
 * \f]
 *
 * @note
 * - The interpretation of `eta` and the sign conventions depend on how you provide `i` and `n`.
 *   In typical shading code, `i` points *toward* the surface and `n` points *outward*.
 * - If `n` is not normalized, results will be incorrect.
 * - Returning `i - i` is used to yield a zero vector expression with matching shape/type.
 */
template <VectorExpressionType EI, VectorExpressionType EN>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
refract(const EI& i, const EN& n, expr_value_t<EI> eta) noexcept {
    using T = std::common_type_t<expr_value_t<EI>, expr_value_t<EN>>;

    const T d = dot(i, n);                         // Cosine-like term (depends on convention).
    const T k = T(1) - eta * eta * (T(1) - d * d); // Discriminant under sqrt.

    if (k < T(0)) {
        // Total internal reflection: return a zero vector expression.
        return i - i;
    }

    // t = eta*i - n*(eta*d + sqrt(k))
    return i * eta - n * (eta * d + static_cast<T>(std::sqrt(k)));
}

} // namespace atlas::math
