#pragma once

/**
 * @file test_utils.h
 * @brief Declares reusable test helpers, dummy runtime objects, and optional Vizkit test fixtures.
 *
 * @details
 * This header provides a collection of small utilities used across Atlas test code.
 * It includes:
 * - scalar and vector comparison helpers,
 * - finite-value validation helpers,
 * - host-side copies of device ranges and buffers,
 * - dummy codec and measure implementations for runtime orchestration tests,
 * - convenience factories for common geometry, domain, sync, fluid, source, sink,
 *   collider, and unit test objects,
 * - optional Vizkit-specific test fixtures when `ATLAS_ENABLE_VIZKIT` is enabled.
 *
 * ## Purpose
 * These utilities reduce boilerplate in tests by centralizing common patterns such as:
 * - approximate floating-point comparison,
 * - copying device buffers to host vectors,
 * - constructing small canonical geometry objects,
 * - creating fully configured runtime objects suitable for integration tests,
 * - providing simple mock-like implementations of abstract interfaces.
 *
 * ## Design notes
 * - Most helpers are intentionally lightweight and header-only.
 * - Many factories return fully initialized objects with stable, deterministic defaults.
 * - Approximate comparisons are designed primarily for floating-point tests.
 * - Dummy runtime classes record whether specific methods were called, making them
 *   useful for orchestration and pipeline verification.
 *
 * ## Conditional Vizkit support
 * When `ATLAS_ENABLE_VIZKIT` is defined, this header also exposes:
 * - unit factories tailored for visualization tests,
 * - a dummy Vizkit layer,
 * - a test geometry layer exposing otherwise protected synchronization helpers.
 *
 * ---
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace atlas::test {

/**
 * @brief Compare two scalar values for approximate equality.
 *
 * @details
 * For floating-point types, the comparison uses an absolute tolerance:
 * \f[
 * |a - b| \le \epsilon
 * \f]
 *
 * For non-floating-point types, the comparison falls back to exact equality.
 *
 * @param a First value.
 * @param b Second value.
 * @param eps Allowed absolute tolerance for floating-point comparisons.
 * @return `true` if the values are considered equal; otherwise `false`.
 *
 * @tparam T Scalar type.
 */
template <typename T>
static ATLAS_FORCE_INLINE bool
near(T a, T b, T eps) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= eps;
    } else {
        return a == b;
    }
}

/**
 * @brief Return whether every component of a vector-like object is finite.
 *
 * @details
 * For floating-point component types, each element is checked with `std::isfinite`.
 * For non-floating-point component types, the function returns `true`.
 *
 * The vector-like type is expected to provide:
 * - `operator[]`,
 * - a static `size()` member.
 *
 * @param v Vector-like object to validate.
 * @return `true` if all floating-point components are finite; otherwise `false`.
 *
 * @tparam V Vector-like type.
 */
template <typename V>
static ATLAS_FORCE_INLINE bool
is_finite_vec(const V& v) {
    using T = decltype(v[0]);

    if constexpr (std::is_floating_point_v<T>) {
        for (std::size_t i = 0; i < V::size(); ++i) {
            if (!std::isfinite(static_cast<double>(v[i]))) return false;
        }
    }

    return true;
}

/**
 * @brief Compare two Atlas vectors component-wise using an absolute tolerance.
 *
 * @param a First vector.
 * @param b Second vector.
 * @param eps Allowed per-component absolute tolerance.
 * @return `true` if all components are approximately equal; otherwise `false`.
 *
 * @tparam T Scalar type.
 * @tparam N Vector dimension.
 */
template <typename T, std::size_t N>
static ATLAS_FORCE_INLINE bool
vec_near(const atlas::Vector<T, N>& a,
         const atlas::Vector<T, N>& b,
         T eps) {
    for (std::size_t i = 0; i < N; ++i) {
        if (!near<T>(a[i], b[i], eps)) return false;
    }

    return true;
}

} // namespace atlas::test
