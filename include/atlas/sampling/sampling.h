#pragma once

/**
 * @file sampling.h
 * @brief Declares sampling utility functions for random directions, hemispheres, and hashed scalar generation.
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/shuffle/shuffle_operator.h>
#include <cmath>
#include <cstdint>

namespace atlas::sampling {

/**
 * @brief Generates two independent standard normal random samples.
 *
 * This function draws two scalar samples from the standard normal distribution
 * with mean 0 and variance 1. The implementation uses two uniform random
 * samples and applies one Box-Muller transform.
 *
 * @tparam T Floating-point scalar type.
 * @param engine Random engine used to generate uniform samples.
 * @param first First output sample distributed approximately as N(0, 1).
 * @param second Second output sample distributed approximately as N(0, 1).
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
generate_standard_normal_pair(atlas::default_random_engine<T>& engine,
                              T& first,
                              T& second) {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = std::max(dist(engine), static_cast<T>(eps));

    const T u2 = dist(engine);

    const T r = atlas::math::sqrt_nonnegative(T(-2) * std::log(u1));

    const T theta = T(2) * static_cast<T>(atlas::pi) * u2;

    first  = r * std::cos(theta);
    second = r * std::sin(theta);
}

/**
 * @brief Generates a standard normal random sample.
 *
 * @tparam T Floating-point scalar type.
 * @param engine Random engine used to generate uniform samples.
 * @return A scalar sample distributed approximately as N(0, 1).
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
generate_standard_normal(atlas::default_random_engine<T>& engine) {
    T first {};
    T second {};
    generate_standard_normal_pair(engine, first, second);
    return first;
}

/**
 * @brief Builds an orthonormal basis around a normal vector.
 *
 * Given a normal vector @p n, this function computes two perpendicular unit
 * vectors @p t and @p b such that:
 * - @p t is orthogonal to @p n
 * - @p b is orthogonal to both @p n and @p t
 *
 * The resulting basis can be used to transform samples from local tangent-space
 * coordinates into world-space directions aligned with @p n.
 *
 * @tparam T Floating-point scalar type.
 * @param n Input normal vector.
 * @param t Output tangent vector.
 * @param b Output bitangent vector.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {
    if (!atlas::math::orthonormal_basis(n, t, b)) {
        t = Vector3<T>(T(1), T(0), T(0));
        b = Vector3<T>(T(0), T(1), T(0));
    }
}

/**
 * @brief Samples a direction uniformly over the hemisphere around a normal.
 *
 * The returned direction lies in the hemisphere centered around @p n, with all
 * directions sampled with equal probability over solid angle.
 *
 * The inputs @p u1 and @p u2 are assumed to be uniform random samples in [0, 1).
 *
 * @tparam T Floating-point scalar type.
 * @param n Hemisphere normal direction.
 * @param u1 First uniform random sample in [0, 1).
 * @param u2 Second uniform random sample in [0, 1).
 * @return A unit direction sampled uniformly on the hemisphere defined by @p n.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T phi       = T(2) * static_cast<T>(atlas::pi) * u2;
    const T cos_theta = T(1) - u1;

    return atlas::math::spherical_direction(n, cos_theta, phi);
}

/**
 * @brief Samples a cosine-weighted direction over the hemisphere around a normal.
 *
 * The returned direction lies in the hemisphere centered around @p n, with
 * probability density proportional to the cosine of the angle from the normal.
 *
 * This sampling mode is commonly used for diffuse or Lambertian-style models.
 *
 * The inputs @p u1 and @p u2 are assumed to be uniform random samples in [0, 1).
 *
 * @tparam T Floating-point scalar type.
 * @param n Hemisphere normal direction.
 * @param u1 First uniform random sample in [0, 1).
 * @param u2 Second uniform random sample in [0, 1).
 * @return A unit direction sampled with cosine weighting on the hemisphere defined by @p n.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T phi       = T(2) * static_cast<T>(atlas::pi) * u1;
    const T cos_theta = atlas::math::sqrt_nonnegative(T(1) - u2);

    return atlas::math::spherical_direction(n, cos_theta, phi);
}

/**
 * @brief Samples a random unit vector over the full sphere.
 *
 * The returned direction is uniformly distributed over the unit sphere.
 *
 * @tparam T Floating-point scalar type.
 * @param engine Random engine used to generate uniform samples.
 * @return A unit vector sampled uniformly over the sphere.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_random_unit_vector(atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = dist(engine);
    const T u2 = dist(engine);

    const T cos_theta = T(2) * u1 - T(1);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    return atlas::math::spherical_direction(cos_theta, phi);
}

/**
 * @brief Samples a direction biased around an incoming direction.
 *
 * The returned direction is sampled around @p incoming_direction using a
 * directional concentration parameter @p alpha.
 *
 * Larger values of @p alpha increase alignment with the incoming direction,
 * while non-positive values are clamped to 1 internally.
 *
 * @tparam T Floating-point scalar type.
 * @param incoming_direction Central direction around which to sample.
 * @param alpha Directional concentration parameter.
 * @param engine Random engine used to generate uniform samples.
 * @return A sampled direction biased around @p incoming_direction.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_directional_unit_vector(const Vector3<T>& incoming_direction,
                               T alpha,
                               atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = dist(engine);
    const T u2 = dist(engine);

    alpha = (alpha > T(0)) ? alpha : T(1);

    const T cos_theta = T(2) * std::pow(u1, T(1) / alpha) - T(1);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    return atlas::math::spherical_direction(incoming_direction, cos_theta, phi);
}

/**
 * @brief Computes the number of uniformly spaced samples along one axis.
 *
 * This function returns the number of samples required to cover the interval
 * [@p lower, @p upper] inclusively with spacing @p spacing.
 *
 * Invalid inputs such as non-finite values, non-positive spacing, or negative
 * interval extent result in 0.
 *
 * @tparam T Floating-point scalar type.
 * @param lower Lower bound of the interval.
 * @param upper Upper bound of the interval.
 * @param spacing Distance between adjacent samples.
 * @return Number of axis-aligned samples, or 0 for invalid input.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(T lower, T upper, T spacing) noexcept {
    if (!atlas::math::isfinite(lower) || !atlas::math::isfinite(upper) || !atlas::math::isfinite(spacing)
        || spacing <= T(0))
        return 0;

    const T extent = upper - lower;
    if (extent < T(0)) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

/**
 * @brief Generates a deterministic hashed scalar sample in the interval [0, 1).
 *
 * This function combines the components of @p seed and the supplied @p salt
 * into a deterministic hash-like phase value, applies a sine-based transform,
 * and maps the result into the unit interval.
 *
 * Unlike engine-based random sampling, this function is deterministic for the
 * same input seed and salt pair.
 *
 * @tparam T Floating-point scalar type.
 * @param seed Seed vector used to derive the hashed sample.
 * @param salt Additional scalar salt used to decorrelate samples.
 * @return Deterministic scalar value in the interval [0, 1).
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sample_hashed_unit_interval(const Vector3<T>& seed, const T salt) noexcept {

    const T phase = seed.x * T(atlas::seed::RANDOM_HASH_PHASE_COEFF_X)
        + seed.y * T(atlas::seed::RANDOM_HASH_PHASE_COEFF_Y)
        + seed.z * T(atlas::seed::RANDOM_HASH_PHASE_COEFF_Z)
        + salt;

    const T value = std::sin(phase) * T(atlas::seed::RANDOM_HASH_VALUE_SCALE);

    return value - std::floor(value);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sample_hashed_unit_interval(const int index, const std::uint64_t seed) noexcept {
    const std::uint64_t value = atlas::ShuffleOperator {}(index, seed);
    return static_cast<T>(value >> atlas::seed::RANDOM_HASH_UNIT_INTERVAL_SHIFT)
        * T(atlas::seed::RANDOM_HASH_UNIT_INTERVAL_SCALE);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_hashed_index(const int index,
                    const int upper_bound,
                    const std::uint64_t seed) noexcept {
    if (upper_bound <= 0) {
        return 0;
    }

    const std::uint64_t value = atlas::ShuffleOperator {}(index, seed);
    return static_cast<int>(value % static_cast<std::uint64_t>(upper_bound));
}

} // namespace atlas::sampling
