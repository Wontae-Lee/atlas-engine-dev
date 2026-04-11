#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/random/uniform_real_distribution.h>
#include <cmath>

namespace atlas::sampling {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
generate_standard_normal(atlas::default_random_engine<T>& engine) {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    const T u1 = std::max(dist(engine), static_cast<T>(eps));
    const T u2 = dist(engine);

    const T r     = std::sqrt(T(-2) * std::log(u1));
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
    const T phi    = two_pi * u2;

    const T cos_theta   = T(1) - u1;
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;
    const T sin_theta   = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    const T cos_phi = std::cos(phi);
    const T sin_phi = std::sin(phi);
    const T x       = sin_theta * cos_phi;
    const T y       = sin_theta * sin_phi;
    const T z       = cos_theta;

    Vector3<T> t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);
    return x * t + y * b + z * n;
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {
    const T two_pi = T(2) * M_PI;
    const T phi    = two_pi * u1;

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
    const T phi       = T(2) * static_cast<T>(atlas::pi) * u2;

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
    alpha      = (alpha > T(0)) ? alpha : T(1);

    const T cos_theta = T(2) * std::pow(u1, T(1) / alpha) - T(1);
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;
    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);
    const T phi       = T(2) * static_cast<T>(atlas::pi) * u2;

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

template <typename T, typename GeometryOperator, typename Predicate>
ATLAS_HOST void
sample_spawn_grid(DeviceBuffer<Vector3<T>>& particles,
                  const GeometryOperator& query,
                  T spacing,
                  T tolerance,
                  Predicate predicate) {
    particles.clear();

    if (!std::isfinite(spacing) || spacing <= T(0)) return;

    const auto bounds = query.bound();

    const auto& lower = bounds.lower_corner;
    const auto& upper = bounds.upper_corner;

    const int nx = sample_axis_count(lower.x, upper.x, spacing);
    const int ny = sample_axis_count(lower.y, upper.y, spacing);
    const int nz = sample_axis_count(lower.z, upper.z, spacing);

    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const std::size_t total_particles =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    particles.resize(total_particles);
    DeviceBuffer<int> keep_mask(total_particles, 0);

    auto* sampled_particles_ptr = atlas::raw_pointer_cast(particles.data());
    auto* keep_mask_ptr         = atlas::raw_pointer_cast(keep_mask.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0), total_particles, [&](std::size_t linear_index) {
            const int ix = static_cast<int>(linear_index % static_cast<std::size_t>(nx));
            const int iy = static_cast<int>((linear_index / static_cast<std::size_t>(nx))
                                            % static_cast<std::size_t>(ny));
            const int iz = static_cast<int>(linear_index
                                            / (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)));

            const Vector3<T> sample(
                lower.x + static_cast<T>(ix) * spacing,
                lower.y + static_cast<T>(iy) * spacing,
                lower.z + static_cast<T>(iz) * spacing);

            sampled_particles_ptr[linear_index] = sample;
            keep_mask_ptr[linear_index]         = predicate(query, sample, tolerance) ? 1 : 0;
        });

    DeviceBuffer<std::size_t> accepted_offsets(total_particles);
    atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        keep_mask.begin(),
        keep_mask.end(),
        accepted_offsets.begin(),
        static_cast<std::size_t>(0));

    int last_keep = 0;
    std::size_t accepted_count = 0;
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(keep_mask.data()) + (total_particles - 1),
        &last_keep,
        1);
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(accepted_offsets.data()) + (total_particles - 1),
        &accepted_count,
        1);
    accepted_count += static_cast<std::size_t>(last_keep);

    if (accepted_count == 0) {
        particles.clear();
        return;
    }

    DeviceBuffer<Vector3<T>> accepted_particles(accepted_count);
    auto* accepted_particles_ptr = atlas::raw_pointer_cast(accepted_particles.data());
    const auto* accepted_offsets_ptr = atlas::raw_pointer_cast(accepted_offsets.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0), total_particles, [=](std::size_t linear_index) {
            if (keep_mask_ptr[linear_index] != 0) {
                accepted_particles_ptr[accepted_offsets_ptr[linear_index]] = sampled_particles_ptr[linear_index];
            }
        });

    particles = std::move(accepted_particles);
}

}
