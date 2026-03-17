#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <cmath>

/**
 * @file sampling.h
 * @brief Device-side helpers for building an orthonormal basis and sampling a hemisphere.
 *
 * @details
 * This header provides small, CUDA/device-friendly routines commonly used in Monte Carlo
 * rendering / stochastic simulation:
 *
 * - `build_orthonormal_basis(n, t, b)`
 *   Constructs a tangent (`t`) and bitangent (`b`) such that `{t, b, n}` forms an orthonormal basis.
 *
 * - `sample_uniform_hemisphere(n, u1, u2)`
 *   Generates a direction uniformly distributed over the hemisphere oriented around `n`.
 *
 * - `sample_cosine_hemisphere(n, u1, u2)`
 *   Generates a direction distributed proportionally to `cos(theta)` over the hemisphere oriented around `n`
 *   (cosine-weighted sampling), often used for diffuse BRDF importance sampling.
 *
 * All functions are marked `ATLAS_DEVICE` and are intended to be callable inside GPU kernels.
 *
 * @note
 * - These routines assume `n` is a valid normal direction. For best results, pass a normalized `n`
 *   (unit length). If `n` is not normalized, the returned sample direction may not be unit length
 *   and the basis may be distorted.
 * - `M_PI` is used; ensure your build defines it appropriately or replace with your own constant
 *   if needed for portability.
 */

namespace atlas::random {

/**
 * @brief Builds an orthonormal tangent frame from a normal vector.
 *
 * @details
 * Given a (typically unit) normal `n`, this function constructs two perpendicular vectors:
 * - `t` (tangent)
 * - `b` (bitangent)
 *
 * such that:
 * - `t` is perpendicular to `n`
 * - `b = cross(n, t)` (also perpendicular to both)
 * - `{t, b, n}` is (approximately) orthonormal
 *
 * The implementation selects a helper axis based on the dominant component of `n` to avoid
 * numerical issues when `n` is nearly aligned with a coordinate axis.
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 * @param n Input normal direction (preferably normalized).
 * @param t Output tangent vector.
 * @param b Output bitangent vector.
 *
 * @note
 * - If `n` is extremely close to zero length, normalization of `t` may be unstable.
 * - `t` is explicitly normalized; `b` is computed from a cross product and will be unit length
 *   if both `n` and `t` are unit and orthogonal.
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {
    // Choose a vector that is not parallel to n to form a stable tangent direction.
    // If |nx| > |nz|, use (-ny, nx, 0); otherwise use (0, -nz, ny).
    if (std::abs(n.x) > std::abs(n.z)) {
        t = Vector3<T>(-n.y, n.x, T(0));
    } else {
        t = Vector3<T>(T(0), -n.z, n.y);
    }

    // Normalize the tangent so the basis becomes orthonormal (assuming n is unit length).
    t = math::normalize(t);

    // Bitangent is perpendicular to both n and t (right-handed frame).
    b = math::cross(n, t);
}

/**
 * @brief Samples a direction uniformly over the hemisphere oriented around `n`.
 *
 * @details
 * Generates a hemisphere direction using two uniform random variables `u1`, `u2` in [0,1).
 * The sampling is uniform with respect to solid angle over the hemisphere:
 * \f[
 *   p(\omega) = \frac{1}{2\pi} \quad \text{for } \omega \cdot n \ge 0.
 * \f]
 *
 * Mapping used:
 * - \f$\phi = 2\pi u_2\f$
 * - \f$\cos\theta = 1 - u_1\f$  (uniform in cos(theta) over [0,1])
 * - \f$\sin\theta = \sqrt{1 - \cos^2\theta}\f$
 *
 * The local sample `(x, y, z)` is then rotated into world space using an orthonormal basis
 * derived from `n`.
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 * @param n Hemisphere normal direction (preferably normalized).
 * @param u1 Uniform random variable in [0, 1).
 * @param u2 Uniform random variable in [0, 1).
 * @return A direction vector on the hemisphere oriented around `n`.
 *
 * @note
 * - Returned vector is unit length if `n` is unit length and the basis is orthonormal.
 * - This routine does not clamp `u1`/`u2`; caller should supply values in [0, 1).
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {
    // Azimuthal angle in [0, 2*pi).
    const T two_pi = T(2) * M_PI;
    const T phi    = two_pi * u2;

    // Uniform hemisphere sampling: cos(theta) is uniform in [0, 1].
    const T cos_theta   = T(1) - u1;
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;
    const T sin_theta   = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    // Convert spherical coordinates to local Cartesian coordinates.
    const T cos_phi = std::cos(phi);
    const T sin_phi = std::sin(phi);
    const T x       = sin_theta * cos_phi;
    const T y       = sin_theta * sin_phi;
    const T z       = cos_theta;

    // Build local frame around n and rotate local direction into world space.
    Vector3<T> t, b;
    atlas::random::build_orthonormal_basis(n, t, b);
    return x * t + y * b + z * n;
}

/**
 * @brief Samples a cosine-weighted direction over the hemisphere oriented around `n`.
 *
 * @details
 * Cosine-weighted hemisphere sampling is commonly used for Lambertian/diffuse importance sampling.
 * The distribution is proportional to \f$\cos\theta\f$:
 * \f[
 *   p(\omega) = \frac{\cos\theta}{\pi} \quad \text{for } \omega \cdot n \ge 0.
 * \f]
 *
 * Mapping used:
 * - \f$\phi = 2\pi u_1\f$
 * - \f$\cos\theta = \sqrt{1 - u_2}\f$
 * - \f$\sin\theta = \sqrt{u_2}\f$
 *
 * The local sample `(x, y, z)` is then rotated into world space using an orthonormal basis
 * derived from `n`.
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 * @param n Hemisphere normal direction (preferably normalized).
 * @param u1 Uniform random variable in [0, 1).
 * @param u2 Uniform random variable in [0, 1).
 * @return A cosine-weighted direction on the hemisphere oriented around `n`.
 *
 * @note
 * - Returned vector is unit length if `n` is unit length and the basis is orthonormal.
 * - This routine does not compute or return the PDF. If you need it:
 *   - Uniform hemisphere: `pdf = 1 / (2*pi)`
 *   - Cosine hemisphere:  `pdf = max(0, dot(n, w)) / pi`
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {
    const T two_pi = T(2) * M_PI;
    const T phi    = two_pi * u1;

    // Cosine-weighted sampling: z = cos(theta) = sqrt(1 - u2).
    const T cos_theta = std::sqrt(T(1) - u2);
    const T sin_theta = std::sqrt(u2);

    const T cos_phi = std::cos(phi);
    const T sin_phi = std::sin(phi);

    // Local coordinates (x,y) on the disk, lifted to hemisphere by z.
    const T x = sin_theta * cos_phi;
    const T y = sin_theta * sin_phi;
    const T z = cos_theta;

    Vector3<T> t, b;
    atlas::random::build_orthonormal_basis(n, t, b);
    return x * t + y * b + z * n;
}

} // namespace atlas::random

namespace atlas::sampling {

/**
 * @brief Computes the number of regular grid samples on one axis.
 *
 * @tparam T Floating-point scalar type.
 * @param lower Lower bound of the axis interval.
 * @param upper Upper bound of the axis interval.
 * @param spacing Uniform grid spacing.
 * @return Number of samples including both interval endpoints when reachable by stepping.
 */
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE int
sample_axis_count(T lower, T upper, T spacing) noexcept {
    if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(spacing) || spacing <= T(0))
        return 0;

    const T extent = upper - lower;
    if (extent < T(0)) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

/**
 * @brief Builds particles from a regular grid clipped by a query predicate.
 *
 * @details
 * The routine:
 * - enumerates all grid samples inside `query.bound()`
 * - evaluates `predicate(query, sample, tolerance)` in parallel
 * - compacts accepted samples into `particles` using exclusive scan
 *
 * @tparam T Floating-point scalar type.
 * @tparam Predicate Callable returning `bool` for `(query, sample, tolerance)`.
 * @param particles Output particle buffer.
 * @param query Query operator providing bounds and point classification.
 * @param spacing Uniform grid spacing.
 * @param tolerance Classification tolerance forwarded to `predicate`.
 * @param predicate Point acceptance predicate.
 */
template <typename T, typename QueryOperator, typename Predicate>
ATLAS_HOST void
sample_spawn_grid(DeviceBuffer<Vector3<T>>& particles,
                  const QueryOperator& query,
                  T spacing,
                  T tolerance,
                  Predicate predicate) {
    particles.clear();

    if (!query.is_valid() || !std::isfinite(spacing) || spacing <= T(0)) return;

    const auto bounds = query.bound();
    if (!bounds.is_valid()) return;

    const auto& lower = bounds.lower_corner;
    const auto& upper = bounds.upper_corner;

    const int nx = sample_axis_count(lower.x, upper.x, spacing);
    const int ny = sample_axis_count(lower.y, upper.y, spacing);
    const int nz = sample_axis_count(lower.z, upper.z, spacing);

    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const std::size_t total_candidates =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    if (total_candidates == 0) return;

    DeviceBuffer<Vector3<T>> candidates(total_candidates);
    DeviceBuffer<int> keep_mask(total_candidates, 0);
    DeviceBuffer<int> offsets(total_candidates, 0);

    Vector3<T>* candidates_ptr = atlas::raw_pointer_cast(candidates.data());
    int* keep_mask_ptr         = atlas::raw_pointer_cast(keep_mask.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(total_candidates),
        [=] ATLAS_ALL_DEVICE(const int index) {
            const int plane = nx * ny;
            const int iz    = index / plane;
            const int rem   = index - iz * plane;
            const int iy    = rem / nx;
            const int ix    = rem - iy * nx;

            const Vector3<T> sample(
                lower.x + static_cast<T>(ix) * spacing,
                lower.y + static_cast<T>(iy) * spacing,
                lower.z + static_cast<T>(iz) * spacing);

            candidates_ptr[index] = sample;
            keep_mask_ptr[index]  = predicate(query, sample, tolerance) ? 1 : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        keep_mask.begin(),
        keep_mask.end(),
        offsets.begin(),
        0);

    const int kept = offsets.back() + keep_mask.back();
    if (kept <= 0) return;

    particles.resize(static_cast<std::size_t>(kept));

    const Vector3<T>* candidate_ptr = atlas::raw_pointer_cast(candidates.data());
    const int* offsets_ptr          = atlas::raw_pointer_cast(offsets.data());
    const int* keep_ptr             = atlas::raw_pointer_cast(keep_mask.data());
    Vector3<T>* particles_ptr       = atlas::raw_pointer_cast(particles.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(total_candidates),
        [=] ATLAS_ALL_DEVICE(const int index) {
            if (!keep_ptr[index]) return;
            particles_ptr[offsets_ptr[index]] = candidate_ptr[index];
        });
}

} // namespace atlas::sampling
