#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
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

    HostBuffer<Vector3<T>> sampled_particles(total_particles);
    HostBuffer<unsigned char> keep_mask(total_particles, 0);

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

            sampled_particles[linear_index] = sample;
            keep_mask[linear_index]         = predicate(query, sample, tolerance) ? 1 : 0;
        });

    HostBuffer<Vector3<T>> accepted_particles;
    accepted_particles.reserve(total_particles);

    for (std::size_t i = 0; i < total_particles; ++i) {
        if (keep_mask[i] != 0) {
            accepted_particles.push_back(sampled_particles[i]);
        }
    }

    if (accepted_particles.empty()) return;

    particles = DeviceBuffer<Vector3<T>>(accepted_particles.begin(), accepted_particles.end());
}

}
