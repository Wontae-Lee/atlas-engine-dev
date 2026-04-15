#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <cmath>

namespace atlas::sampling {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
generate_standard_normal(atlas::default_random_engine<T>& engine) {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = std::max(dist(engine), static_cast<T>(eps));

    const T u2 = dist(engine);

    const T r = std::sqrt(T(-2) * std::log(u1));

    const T theta = T(2) * static_cast<T>(atlas::pi) * u2;

    return r * std::cos(theta);
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {

    if (std::abs(n.x) > std::abs(n.z)) {
        t = Vector3<T>(-n.y, n.x, T(0));
    } else {
        t = Vector3<T>(T(0), -n.z, n.y);
    }

    t = math::normalize(t);

    b = math::cross(n, t);
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T two_pi = T(2) * M_PI;

    const T phi = two_pi * u2;

    const T cos_theta = T(1) - u1;

    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    const T cos_phi = std::cos(phi);

    const T sin_phi = std::sin(phi);

    const T x = sin_theta * cos_phi;
    const T y = sin_theta * sin_phi;
    const T z = cos_theta;

    Vector3<T> t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    return x * t + y * b + z * n;
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {

    const T two_pi = T(2) * M_PI;

    const T phi = two_pi * u1;

    const T cos_theta = std::sqrt(T(1) - u2);

    const T sin_theta = std::sqrt(u2);

    const T cos_phi = std::cos(phi);
    const T sin_phi = std::sin(phi);

    const T x = sin_theta * cos_phi;
    const T y = sin_theta * sin_phi;
    const T z = cos_theta;

    Vector3<T> t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    return x * t + y * b + z * n;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_random_unit_vector(atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = dist(engine);
    const T u2 = dist(engine);

    const T cos_theta = T(2) * u1 - T(1);

    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    return Vector3<T>(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
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

    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    Vector3<T> tangent;
    Vector3<T> bitangent;
    atlas::sampling::build_orthonormal_basis(incoming_direction, tangent, bitangent);

    return tangent * (sin_theta * std::cos(phi))
        + bitangent * (sin_theta * std::sin(phi))
        + incoming_direction * cos_theta;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(T lower, T upper, T spacing) noexcept {
    if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(spacing) || spacing <= T(0))
        return 0;

    const T extent = upper - lower;
    if (extent < T(0)) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sample_hashed_unit_interval(const Vector3<T>& seed, const T salt) noexcept {

    const T phase = seed.x * T(atlas::seed::random_hash_phase_coeff_x)
        + seed.y * T(atlas::seed::random_hash_phase_coeff_y)
        + seed.z * T(atlas::seed::random_hash_phase_coeff_z)
        + salt;

    const T value = std::sin(phase) * T(atlas::seed::random_hash_value_scale);

    return value - std::floor(value);
}

}