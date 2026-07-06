#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/shuffle/shuffle.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace atlas {

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
generate_standard_normal_pair(atlas::default_random_engine& engine,
                              float& first,
                              float& second) {
    atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Compare against `eps` by value (not via std::max's `const float&`, which
    // would odr-use the namespace-scope constexpr and leave it undefined in
    // device code).
    const float sampled_u1 = dist(engine);
    const float u1         = sampled_u1 > eps ? sampled_u1 : eps;

    const float u2 = dist(engine);

    const float r = atlas::sqrt_nonnegative(-2.0f * std::log(u1));

    const float theta = 2.0f * atlas::pi * u2;

    first  = r * std::cos(theta);
    second = r * std::sin(theta);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
generate_standard_normal(atlas::default_random_engine& engine) {
    float first {};
    float second {};
    generate_standard_normal_pair(engine, first, second);
    return first;
}

ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
sample_uniform_vector(atlas::default_random_engine& engine,
                      const float min_value,
                      const float max_value) {
    atlas::uniform_real_distribution<float> distribution(min_value, max_value);
    return Vector3(
        distribution(engine),
        distribution(engine),
        distribution(engine));
}

ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
sample_normal_vector(atlas::default_random_engine& engine,
                     const float sigma) {
    float x {};
    float y {};
    float z {};
    float unused {};
    atlas::generate_standard_normal_pair(engine, x, y);
    atlas::generate_standard_normal_pair(engine, z, unused);
    return Vector3(
        sigma * x,
        sigma * y,
        sigma * z);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3& n,
                        Vector3& t,
                        Vector3& b) {
    if (!atlas::orthonormal_basis(n, t, b)) {
        t = Vector3(1.0f, 0.0f, 0.0f);
        b = Vector3(0.0f, 1.0f, 0.0f);
    }
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
sample_uniform_hemisphere(const Vector3& n, const float u1, const float u2) {

    const float phi       = 2.0f * atlas::pi * u2;
    const float cos_theta = 1.0f - u1;

    return atlas::spherical_direction(n, cos_theta, phi);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
sample_cosine_hemisphere(const Vector3& n, const float u1, const float u2) {

    const float phi       = 2.0f * atlas::pi * u1;
    const float cos_theta = atlas::sqrt_nonnegative(1.0f - u2);

    return atlas::spherical_direction(n, cos_theta, phi);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
sample_random_unit_vector(atlas::default_random_engine& engine) noexcept {
    atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float u1 = dist(engine);
    const float u2 = dist(engine);

    const float cos_theta = 2.0f * u1 - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    return atlas::spherical_direction(cos_theta, phi);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
sample_directional_unit_vector(const Vector3& incoming_direction,
                               float alpha,
                               atlas::default_random_engine& engine) noexcept {
    atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float u1 = dist(engine);
    const float u2 = dist(engine);

    alpha = (alpha > 0.0f) ? alpha : 1.0f;

    const float cos_theta = 2.0f * std::pow(u1, 1.0f / alpha) - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    return atlas::spherical_direction(incoming_direction, cos_theta, phi);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(const float lower, const float upper, const float spacing) noexcept {
    if (!atlas::isfinite(lower) || !atlas::isfinite(upper) || !atlas::isfinite(spacing)
        || spacing <= 0.0f)
        return 0;

    const float extent = upper - lower;
    if (extent < 0.0f) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sample_hashed_unit_interval(const Vector3& seed, const float salt) noexcept {

    const float phase = seed.x * atlas::RANDOM_HASH_PHASE_COEFF_X
        + seed.y * atlas::RANDOM_HASH_PHASE_COEFF_Y
        + seed.z * atlas::RANDOM_HASH_PHASE_COEFF_Z
        + salt;

    const float value = std::sin(phase) * atlas::RANDOM_HASH_VALUE_SCALE;

    return value - std::floor(value);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sample_hashed_unit_interval(const int index, const std::uint64_t seed) noexcept {
    const std::uint64_t value = atlas::Shuffle {}(index, seed);
    return static_cast<float>(value >> atlas::RANDOM_HASH_UNIT_INTERVAL_SHIFT)
        * atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_hashed_index(const int index,
                    const int upper_bound,
                    const std::uint64_t seed) noexcept {
    if (upper_bound <= 0) {
        return 0;
    }

    const std::uint64_t value = atlas::Shuffle {}(index, seed);
    return static_cast<int>(value % static_cast<std::uint64_t>(upper_bound));
}

}
