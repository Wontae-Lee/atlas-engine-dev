#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/shuffle/shuffle_operator.h>
#include <cmath>
#include <cstdint>

namespace atlas {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
generate_standard_normal_pair(atlas::default_random_engine<T>& engine,
                              T& first,
                              T& second) {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = std::max(dist(engine), static_cast<T>(eps));

    const T u2 = dist(engine);

    const T r = atlas::sqrt_nonnegative(T(-2) * std::log(u1));

    const T theta = T(2) * static_cast<T>(atlas::pi) * u2;

    first  = r * std::cos(theta);
    second = r * std::sin(theta);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
generate_standard_normal(atlas::default_random_engine<T>& engine) {
    T first {};
    T second {};
    generate_standard_normal_pair(engine, first, second);
    return first;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_vector(atlas::default_random_engine<T>& engine,
                      const T min_value,
                      const T max_value) {
    atlas::uniform_real_distribution<T> distribution(min_value, max_value);
    return Vector3<T>(
        distribution(engine),
        distribution(engine),
        distribution(engine));
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
sample_normal_vector(atlas::default_random_engine<T>& engine,
                     const T sigma) {
    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::generate_standard_normal_pair<T>(engine, x, y);
    atlas::generate_standard_normal_pair<T>(engine, z, unused);
    return Vector3<T>(
        sigma * x,
        sigma * y,
        sigma * z);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {
    if (!atlas::orthonormal_basis(n, t, b)) {
        t = Vector3<T>(T(1), T(0), T(0));
        b = Vector3<T>(T(0), T(1), T(0));
    }
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T phi       = T(2) * static_cast<T>(atlas::pi) * u2;
    const T cos_theta = T(1) - u1;

    return atlas::spherical_direction(n, cos_theta, phi);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T phi       = T(2) * static_cast<T>(atlas::pi) * u1;
    const T cos_theta = atlas::sqrt_nonnegative(T(1) - u2);

    return atlas::spherical_direction(n, cos_theta, phi);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_random_unit_vector(atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = dist(engine);
    const T u2 = dist(engine);

    const T cos_theta = T(2) * u1 - T(1);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    return atlas::spherical_direction(cos_theta, phi);
}

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

    return atlas::spherical_direction(incoming_direction, cos_theta, phi);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(T lower, T upper, T spacing) noexcept {
    if (!atlas::isfinite(lower) || !atlas::isfinite(upper) || !atlas::isfinite(spacing)
        || spacing <= T(0))
        return 0;

    const T extent = upper - lower;
    if (extent < T(0)) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sample_hashed_unit_interval(const Vector3<T>& seed, const T salt) noexcept {

    const T phase = seed.x * T(atlas::RANDOM_HASH_PHASE_COEFF_X)
        + seed.y * T(atlas::RANDOM_HASH_PHASE_COEFF_Y)
        + seed.z * T(atlas::RANDOM_HASH_PHASE_COEFF_Z)
        + salt;

    const T value = std::sin(phase) * T(atlas::RANDOM_HASH_VALUE_SCALE);

    return value - std::floor(value);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sample_hashed_unit_interval(const int index, const std::uint64_t seed) noexcept {
    const std::uint64_t value = atlas::ShuffleOperator {}(index, seed);
    return static_cast<T>(value >> atlas::RANDOM_HASH_UNIT_INTERVAL_SHIFT)
        * T(atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE);
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

}