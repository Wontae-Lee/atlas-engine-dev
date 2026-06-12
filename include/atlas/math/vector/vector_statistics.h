#pragma once

#include <atlas/math/vector/vector_expression.h>
#include <atlas/math/vector/vector_reductions.h>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @file vector_statistics.h
 * @brief Statistical utilities for vector expressions (variance, covariance, correlation, normalization).
 *
 * @details
 * This header provides a set of common statistical functions operating on Atlas vector
 * expression templates. Functions fall into two categories:
 *
 * 1) **Scalar reductions** (immediate evaluation):
 *    - variance (population / sample), stddev (population / sample)
 *    - covariance (population / sample), correlation
 *    - skewness, kurtosis (excess), logsumexp
 *
 * 2) **Lane-wise transforms** (lazy evaluation):
 *    - softmax (returns a unary expression node)
 *    - zscore (returns a unary expression node; safe behavior for zero-variance input)
 *    - minmax_scale (returns a unary expression node; safe behavior for constant input)
 *
 * Numerical behavior:
 * - Accumulators use `double` for improved precision and reduced catastrophic cancellation.
 * - Returned scalar type is `expr_value_t<E>` (or `common_type` for paired inputs).
 * - Degenerate cases (empty input, n<=1, zero variance, etc.) return zero or a well-defined
 *   fallback to avoid division by zero and NaNs.
 *
 * @note
 * - Some routines assume non-empty input and access `v[0]`. If `n==0`, this is undefined.
 *   For correctness, empty inputs should be handled before accessing the first element.
 * - These utilities depend on `mean(...)`, `dot(...)`, and related reductions from
 *   `atlas/math/vector/reductions.h`.
 */

// ------------------------------------------------------------
// Variance / standard deviation
// ------------------------------------------------------------

/**
 * @brief Computes the population variance of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Population variance \f$\sigma^2\f$ computed as \f$\frac{1}{n}\sum_i (x_i-\mu)^2\f$.
 *
 * @details
 * Uses a two-pass approach:
 * 1) Compute mean \f$\mu\f$
 * 2) Accumulate squared deviations into `M2` (double precision)
 *
 * If `n == 0`, returns 0.
 *
 * @note
 * - Accumulation is done in `double` for better numerical stability.
 * - Uses `ATLAS_UNROLL` to encourage loop unrolling for small fixed sizes.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
variance_population(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();   // Bind to the underlying derived expression (no copy).
    const std::size_t n = v.size(); // Number of lanes/elements.

    if (n == 0) return T(0); // Empty input: define variance as 0.

    const T m = mean(expr); // First pass: compute mean (reduction).

    double M2 = 0.0; // Accumulator for sum of squared deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Compute deviation in double to reduce rounding error.
        const auto d = static_cast<double>(v[i] - m);
        M2 += d * d;
    }

    // Population variance divides by n.
    return static_cast<T>(M2 / static_cast<double>(n));
}

/**
 * @brief Computes the sample variance of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Sample variance \f$s^2\f$ computed as \f$\frac{1}{n-1}\sum_i (x_i-\bar{x})^2\f$.
 *
 * @details
 * Uses a two-pass approach (mean + squared deviations).
 * If `n <= 1`, returns 0 to avoid division by zero.
 *
 * @note
 * - Accumulation is done in `double` for improved precision.
 * - Uses Bessel's correction (divide by `n-1`).
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
variance_sample(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();   // Underlying expression reference.
    const std::size_t n = v.size(); // Number of samples.

    if (n <= 1) return T(0); // Undefined/degenerate sample variance.

    const T m = mean(expr); // Sample mean.

    double M2 = 0.0; // Sum of squared deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = static_cast<double>(v[i] - m);
        M2 += d * d;
    }

    // Sample variance divides by (n-1).
    return static_cast<T>(M2 / static_cast<double>(n - 1));
}

/**
 * @brief Computes the population standard deviation of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Population standard deviation \f$\sigma = \sqrt{\sigma^2}\f$.
 *
 * @note
 * Uses `variance_population()` and computes the square root in `double`,
 * then casts back to the expression value type.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
stddev_population(const E& expr) noexcept {
    using T = expr_value_t<E>;
    // Compute sqrt in double for accuracy, then cast back.
    return static_cast<T>(std::sqrt(static_cast<double>(variance_population(expr))));
}

/**
 * @brief Computes the sample standard deviation of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Sample standard deviation \f$s = \sqrt{s^2}\f$.
 *
 * @note
 * Uses `variance_sample()` and computes the square root in `double`,
 * then casts back to the expression value type.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
stddev_sample(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(static_cast<double>(variance_sample(expr))));
}

// ------------------------------------------------------------
// Covariance / correlation
// ------------------------------------------------------------

/**
 * @brief Computes the population covariance between two expressions.
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return Population covariance \f$\frac{1}{n}\sum_i (x_i-\mu_x)(y_i-\mu_y)\f$.
 *
 * @details
 * Uses a two-pass approach:
 * 1) Compute means \f$\mu_x\f$ and \f$\mu_y\f$
 * 2) Accumulate cross-deviations in double precision
 *
 * If `n == 0`, returns 0.
 *
 * @warning
 * This assumes `x.size() == y.size()`. If sizes differ, behavior is undefined.
 */
template <VectorExpressionType EX, VectorExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
covariance_population(const EX& x, const EY& y) noexcept {
    using T = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;

    const auto& a       = x();      // Underlying expression for x.
    const auto& b       = y();      // Underlying expression for y.
    const std::size_t n = a.size(); // Sample count (taken from x).

    if (n == 0) return T(0); // Empty input: define covariance as 0.

    const T mx = mean(x); // Mean of x.
    const T my = mean(y); // Mean of y.

    double acc = 0.0; // Accumulate cross deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Multiply in expression value type, accumulate in double.
        acc += static_cast<double>((a[i] - mx) * (b[i] - my));
    }

    // Population covariance divides by n.
    return static_cast<T>(acc / static_cast<double>(n));
}

/**
 * @brief Computes the sample covariance between two expressions (Bessel's correction).
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return Sample covariance \f$\frac{1}{n-1}\sum_i (x_i-\bar{x})(y_i-\bar{y})\f$.
 *
 * @details
 * If `n <= 1`, returns 0 to avoid division by zero.
 *
 * @warning
 * This assumes `x.size() == y.size()`. If sizes differ, behavior is undefined.
 */
template <VectorExpressionType EX, VectorExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
covariance_sample(const EX& x, const EY& y) noexcept {
    using T = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;

    const auto& a       = x(); // x expression
    const auto& b       = y(); // y expression
    const std::size_t n = a.size();

    if (n <= 1) return T(0); // Degenerate sample covariance.

    const T mx = mean(x);
    const T my = mean(y);

    double acc = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>((a[i] - mx) * (b[i] - my));
    }

    // Sample covariance divides by (n-1).
    return static_cast<T>(acc / static_cast<double>(n - 1));
}

/**
 * @brief Computes the (sample) Pearson correlation coefficient between two expressions.
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return Correlation \f$\rho = \frac{\mathrm{cov}(x,y)}{s_x s_y}\f$.
 *
 * @details
 * Uses sample standard deviations and sample covariance:
 * - `sx = stddev_sample(x)`
 * - `sy = stddev_sample(y)`
 * - `covariance_sample(x,y)`
 *
 * If either standard deviation is zero, returns 0 to avoid division by zero.
 *
 * @note
 * This returns 0 for degenerate inputs; this is a policy choice that avoids NaNs.
 */
template <VectorExpressionType EX, VectorExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
correlation(const EX& x, const EY& y) noexcept {
    using T = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;

    const T sx = stddev_sample(x); // Sample standard deviation of x.
    const T sy = stddev_sample(y); // Sample standard deviation of y.

    if (sx == T(0) || sy == T(0)) {
        // If either variable is constant, correlation is undefined; return 0 by policy.
        return T(0);
    }

    return covariance_sample(x, y) / (sx * sy);
}

// ------------------------------------------------------------
// Higher moments
// ------------------------------------------------------------

/**
 * @brief Computes the (population) skewness of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Skewness \f$\gamma_1 = \mu_3 / \mu_2^{3/2}\f$.
 *
 * @details
 * Computes central moments:
 * - \f$\mu_2 = \frac{1}{n}\sum (x_i-\mu)^2\f$
 * - \f$\mu_3 = \frac{1}{n}\sum (x_i-\mu)^3\f$
 *
 * If `n == 0` or `mu2 == 0`, returns 0.
 *
 * @note
 * This is the population (moment) skewness, not an unbiased estimator for finite samples.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
skewness(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();
    const std::size_t n = v.size();
    if (n == 0) return T(0);

    const T m = mean(expr); // Mean.

    double M2 = 0.0; // Sum of squared deviations.
    double M3 = 0.0; // Sum of cubed deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d    = static_cast<double>(v[i] - m);
        const double d2 = d * d;
        M2 += d2;
        M3 += d2 * d; // d^3
    }

    if (M2 == 0.0) return T(0); // Constant vector -> zero skewness by policy.

    const double mu2 = M2 / static_cast<double>(n);
    const double mu3 = M3 / static_cast<double>(n);

    // gamma1 = mu3 / mu2^(3/2)
    return static_cast<T>(mu3 / std::pow(mu2, 1.5));
}

/**
 * @brief Computes the excess kurtosis of an expression.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Excess kurtosis \f$\gamma_2 = \mu_4/\mu_2^2 - 3\f$.
 *
 * @details
 * Computes central moments:
 * - \f$\mu_2 = \frac{1}{n}\sum (x_i-\mu)^2\f$
 * - \f$\mu_4 = \frac{1}{n}\sum (x_i-\mu)^4\f$
 *
 * Returns 0 if `n == 0` or `mu2 == 0`.
 *
 * @note
 * This is the population (moment) kurtosis excess, not an unbiased estimator.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
kurtosis_excess(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();
    const std::size_t n = v.size();
    if (n == 0) return T(0);

    const T m = mean(expr);

    double M2 = 0.0; // Sum of squared deviations.
    double M4 = 0.0; // Sum of fourth-power deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d    = static_cast<double>(v[i] - m);
        const double d2 = d * d;
        M2 += d2;
        M4 += d2 * d2; // d^4
    }

    if (M2 == 0.0) return T(0); // Constant vector -> zero excess kurtosis by policy.

    const double mu2 = M2 / static_cast<double>(n);
    const double mu4 = M4 / static_cast<double>(n);

    // gamma2 = mu4/mu2^2 - 3
    return static_cast<T>(mu4 / (mu2 * mu2) - 3.0);
}

// ------------------------------------------------------------
// log-sum-exp / softmax
// ------------------------------------------------------------

/**
 * @brief Computes the log-sum-exp of an expression in a numerically stable way.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return \f$\log\sum_i \exp(x_i)\f$.
 *
 * @details
 * Uses the standard stabilization:
 * \f[
 *   \log\sum_i \exp(x_i) = m + \log\sum_i \exp(x_i - m),
 * \quad m = \max_i x_i.
 * \f]
 *
 * @warning
 * The implementation reads `v[0]` to initialize `mx`. If `n == 0`, behavior is undefined.
 * If empty vectors are possible, add an `if (n == 0) return T(0);` before accessing `v[0]`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
logsumexp(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();
    const std::size_t n = v.size();

    // NOTE: Assumes n > 0. Consider guarding if empty inputs can occur.
    T mx = v[0]; // Initialize max with first element.

    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        // Track maximum for numerical stability.
        if (mx < v[i]) mx = v[i];
    }

    double acc = 0.0; // Accumulate exp(x_i - mx) in double.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += std::exp(static_cast<double>(v[i] - mx));
    }

    // lse = log(acc) + mx
    return static_cast<T>(std::log(acc) + static_cast<double>(mx));
}

/**
 * @brief Computes a lane-wise softmax transform (unnormalized by construction of logsumexp).
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Lazy expression computing \f$\exp(x_i - \text{logsumexp}(x))\f$ for each lane.
 *
 * @details
 * The returned expression node evaluates:
 * \f[
 *   \text{softmax}(x_i) = \exp(x_i - \text{LSE}(x)),
 * \quad \text{LSE}(x) = \log\sum_j \exp(x_j).
 * \f]
 *
 * @note
 * This returns an expression node; evaluation is lazy and per-lane.
 * The scalar `lse` is computed eagerly (reduction).
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
softmax(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const T lse = logsumexp(expr); // Eager reduction (scalar).

    // Functor captured by value inside the unary node.
    struct SoftmaxOp {
        T lse;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        operator()(T v) const noexcept {
            // Compute exp(v - lse) in double for better precision, then cast back.
            return static_cast<T>(std::exp(static_cast<double>(v - lse)));
        }
    };

    return VectorUnaryOperator<T, E, SoftmaxOp>(expr(), SoftmaxOp { lse });
}

// ------------------------------------------------------------
// Standardization / scaling
// ------------------------------------------------------------

/**
 * @brief Computes a lane-wise z-score standardization.
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @return Lazy expression computing \f$(x_i - \bar{x})/s\f$ per lane.
 *
 * @details
 * Uses sample statistics:
 * - mean: `m = mean(expr)`
 * - sample standard deviation: `sd = stddev_sample(expr)`
 *
 * Degenerate case handling:
 * - If `sd == 0`, returns an expression that produces all zeros.
 *
 * @note
 * - Returns a lazy expression node; the mean and stddev are computed eagerly (reductions).
 * - The `sd == 0` check is exact; consider epsilon-based policy at higher level if needed.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
zscore(const E& expr) noexcept {
    using T = expr_value_t<E>;

    const T m  = mean(expr);          // Eager mean.
    const T sd = stddev_sample(expr); // Eager stddev (sample).

    if (sd == T(0)) {
        // Constant vector: define z-scores as 0 by policy.
        struct ZeroOp {
            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
            operator()(T) const noexcept { return T(0); }
        };
        return VectorUnaryOperator<T, E, ZeroOp>(expr(), ZeroOp {});
    }

    // Precompute reciprocal to avoid division per lane.
    struct ZOp {
        T m;
        T inv;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        operator()(T v) const noexcept {
            // (v - mean) * (1/sd)
            return (v - m) * inv;
        }
    };

    return VectorUnaryOperator<T, E, ZOp>(expr(), ZOp { m, T(1) / sd });
}

/**
 * @brief Applies min-max scaling to map values into [a, b].
 *
 * @tparam E Vector expression type.
 * @param expr Input expression.
 * @param a Target lower bound.
 * @param b Target upper bound.
 * @return Lazy expression mapping each lane to `a + (x - vmin) * ((b - a) / (vmax - vmin))`,
 *         or the midpoint `(a+b)/2` when input range is zero.
 *
 * @details
 * Finds `vmin` and `vmax` in a pass, then creates a lane-wise affine transform.
 *
 * Degenerate case handling:
 * - If `range == 0` (all values equal), returns a constant expression producing the midpoint.
 *
 * @warning
 * The implementation reads `v[0]` to initialize `vmin`/`vmax`. If `n == 0`, behavior is undefined.
 * If empty vectors are possible, add an early `if (n == 0)` return a suitable expression or scalar.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
minmax_scale(const E& expr, expr_value_t<E> a, expr_value_t<E> b) noexcept {
    using T = expr_value_t<E>;

    const auto& v       = expr();
    const std::size_t n = v.size();

    // NOTE: Assumes n > 0. Consider guarding if empty inputs can occur.
    T vmin = v[0];
    T vmax = v[0];

    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const T x = v[i];

        // Track min/max in a single pass.
        if (x < vmin) vmin = x;
        if (vmax < x) vmax = x;
    }

    const T range = vmax - vmin;
    if (range == T(0)) {
        // Constant input: return midpoint of [a, b] by policy.
        const T mid = (a + b) * T(0.5);

        struct MidOp {
            T mid;

            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
            operator()(T) const noexcept { return mid; }
        };

        return VectorUnaryOperator<T, E, MidOp>(expr(), MidOp { mid });
    }

    // Affine scaling: a + (x - vmin) * s, where s = (b - a) / range.
    struct ScaleOp {
        T vmin;
        T s;
        T a;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        operator()(T x) const noexcept {
            return a + (x - vmin) * s;
        }
    };

    return VectorUnaryOperator<T, E, ScaleOp>(expr(), ScaleOp { vmin, (b - a) / range, a });
}

} // namespace atlas
