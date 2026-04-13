#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/scan/exclusive_scan.h>
#include <cmath>

namespace atlas::sampling {

/**
 * @brief Generate one scalar sample from the standard normal distribution.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param engine Random engine used as the entropy source.
 * @return One sample distributed approximately as N(0, 1).
 *
 * @details
 * This function uses the Box-Muller transform:
 * @f[
 * r = \sqrt{-2 \ln(u_1)}, \quad \theta = 2\pi u_2
 * @f]
 * @f[
 * x = r \cos(\theta)
 * @f]
 *
 * Two uniform random values in the interval [0, 1] are drawn from
 * @ref atlas::uniform_real_distribution. The first sample is clamped from below
 * by @ref atlas::eps to avoid evaluating `log(0)`.
 *
 * The function is available on both host and device builds through
 * @ref ATLAS_ALL_DEVICE.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
generate_standard_normal(atlas::default_random_engine<T>& engine) {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    /// First uniform sample. Clamped to avoid singularity in `log(u1)`.
    const T u1 = std::max(dist(engine), static_cast<T>(eps));

    /// Second uniform sample used to generate the angular phase.
    const T u2 = dist(engine);

    /// Radial term of the Box-Muller transform.
    const T r = std::sqrt(T(-2) * std::log(u1));

    /// Angular term of the Box-Muller transform.
    const T theta = T(2) * static_cast<T>(atlas::pi) * u2;

    /// Return one of the two independent normal samples.
    return r * std::cos(theta);
}

/**
 * @brief Build an orthonormal basis from a unit-like normal vector.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param n Input direction that defines the local z-axis of the basis.
 * @param[out] t Tangent vector orthogonal to @p n.
 * @param[out] b Bitangent vector orthogonal to both @p n and @p t.
 *
 * @details
 * The function constructs a stable tangent by selecting one of two candidate
 * perpendicular vectors depending on the relative magnitudes of `n.x` and `n.z`.
 * The tangent is then normalized, and the bitangent is computed as:
 * @f[
 * b = n \times t
 * @f]
 *
 * This routine assumes that @p n is already normalized or close enough for the
 * intended sampling use case.
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {
    /// Choose a tangent seed that avoids catastrophic cancellation.
    if (std::abs(n.x) > std::abs(n.z)) {
        t = Vector3<T>(-n.y, n.x, T(0));
    } else {
        t = Vector3<T>(T(0), -n.z, n.y);
    }

    /// Normalize the tangent direction.
    t = math::normalize(t);

    /// Complete the orthonormal frame.
    b = math::cross(n, t);
}

/**
 * @brief Sample a direction uniformly over the hemisphere oriented by a normal.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param n Hemisphere pole / surface normal.
 * @param u1 First uniform random variable in [0, 1].
 * @param u2 Second uniform random variable in [0, 1].
 * @return Unit-like direction expressed in world coordinates.
 *
 * @details
 * The sample is first generated in local coordinates where the hemisphere pole
 * is aligned with the local z-axis, then rotated into world space using an
 * orthonormal basis built from @p n.
 *
 * Local parameterization:
 * - `phi` is uniform in [0, 2π)
 * - `cos(theta)` is uniform in [0, 1]
 *
 * This produces a uniform solid-angle distribution over the hemisphere.
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {
    /// Full azimuthal range.
    const T two_pi = T(2) * M_PI;

    /// Uniform azimuth angle.
    const T phi = two_pi * u2;

    /// Uniform cosine-weight along the hemisphere elevation.
    const T cos_theta = T(1) - u1;

    /// Squared sine term derived from the trigonometric identity.
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    /// Positive sine magnitude.
    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    /// Local x-component.
    const T cos_phi = std::cos(phi);

    /// Local y-component.
    const T sin_phi = std::sin(phi);

    /// Local Cartesian direction components.
    const T x = sin_theta * cos_phi;
    const T y = sin_theta * sin_phi;
    const T z = cos_theta;

    /// Build local frame around the target normal.
    Vector3<T> t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    /// Transform local sample into world coordinates.
    return x * t + y * b + z * n;
}

/**
 * @brief Sample a cosine-weighted direction over the hemisphere oriented by a normal.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param n Hemisphere pole / surface normal.
 * @param u1 First uniform random variable in [0, 1].
 * @param u2 Second uniform random variable in [0, 1].
 * @return Unit-like direction expressed in world coordinates.
 *
 * @details
 * This sampler produces a Lambertian-style distribution over the hemisphere.
 * The local sample is generated using:
 * @f[
 * \phi = 2\pi u_1,\quad
 * \cos\theta = \sqrt{1-u_2},\quad
 * \sin\theta = \sqrt{u_2}
 * @f]
 *
 * The resulting local direction is then rotated into world coordinates using
 * an orthonormal basis built from @p n.
 */
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {
    /// Full azimuthal range.
    const T two_pi = T(2) * M_PI;

    /// Uniform azimuth angle.
    const T phi = two_pi * u1;

    /// Cosine-weighted elevation component.
    const T cos_theta = std::sqrt(T(1) - u2);

    /// Corresponding sine magnitude.
    const T sin_theta = std::sqrt(u2);

    /// Trigonometric azimuth components.
    const T cos_phi = std::cos(phi);
    const T sin_phi = std::sin(phi);

    /// Local Cartesian direction components.
    const T x = sin_theta * cos_phi;
    const T y = sin_theta * sin_phi;
    const T z = cos_theta;

    /// Build local frame around the target normal.
    Vector3<T> t, b;
    atlas::sampling::build_orthonormal_basis(n, t, b);

    /// Transform local sample into world coordinates.
    return x * t + y * b + z * n;
}

/**
 * @brief Sample a random unit vector uniformly over the sphere.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param engine Random engine used to generate the sample.
 * @return Unit-like vector uniformly distributed over the full sphere.
 *
 * @details
 * The construction uses two independent uniform variables:
 * - `u1` determines `cos(theta)` uniformly in [-1, 1]
 * - `u2` determines `phi` uniformly in [0, 2π)
 *
 * This produces an isotropic direction distribution on the sphere.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_random_unit_vector(atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    /// Uniform variates used for polar and azimuthal angles.
    const T u1 = dist(engine);
    const T u2 = dist(engine);

    /// Uniform cosine of the polar angle in [-1, 1].
    const T cos_theta = T(2) * u1 - T(1);

    /// Squared sine from trigonometric identity.
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    /// Positive sine magnitude.
    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    /// Azimuth angle in [0, 2π).
    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    /// Return the sampled spherical direction converted to Cartesian coordinates.
    return Vector3<T>(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
}

/**
 * @brief Sample a unit vector biased around an incoming direction.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param incoming_direction Preferred / central direction.
 * @param alpha Concentration-like shaping parameter.
 * @param engine Random engine used to generate the sample.
 * @return Unit-like direction expressed in world coordinates.
 *
 * @details
 * This routine samples a spherical distribution around @p incoming_direction.
 * Larger values of @p alpha sharpen the distribution around the incoming axis.
 * Non-positive `alpha` is replaced by `1`.
 *
 * The local sample is generated in a basis aligned with the incoming direction,
 * then rotated into world space.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
sample_directional_unit_vector(const Vector3<T>& incoming_direction,
                               T alpha,
                               atlas::default_random_engine<T>& engine) noexcept {
    atlas::uniform_real_distribution<T> dist(T(0), T(1));

    /// Uniform variates for the shaped polar/azimuthal sample.
    const T u1 = dist(engine);
    const T u2 = dist(engine);

    /// Clamp non-positive shaping parameter to a neutral default.
    alpha = (alpha > T(0)) ? alpha : T(1);

    /// Shaped cosine of the polar angle.
    const T cos_theta = T(2) * std::pow(u1, T(1) / alpha) - T(1);

    /// Squared sine from trigonometric identity.
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;

    /// Positive sine magnitude.
    const T sin_theta = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);

    /// Azimuth angle in [0, 2π).
    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    /// Local tangent frame around the incoming direction.
    Vector3<T> tangent;
    Vector3<T> bitangent;
    atlas::sampling::build_orthonormal_basis(incoming_direction, tangent, bitangent);

    /// Transform the local sample into world coordinates.
    return tangent * (sin_theta * std::cos(phi))
        + bitangent * (sin_theta * std::sin(phi))
        + incoming_direction * cos_theta;
}

/**
 * @brief Compute the number of regularly spaced samples along one axis.
 *
 * @tparam T Floating-point scalar type.
 *
 * @param lower Lower bound of the interval.
 * @param upper Upper bound of the interval.
 * @param spacing Desired grid spacing.
 * @return Number of grid points along the axis, or zero when the interval is invalid.
 *
 * @details
 * The count follows:
 * @f[
 * \left\lfloor \frac{upper-lower}{spacing} \right\rfloor + 1
 * @f]
 *
 * The result is zero when:
 * - any input is non-finite,
 * - `spacing <= 0`,
 * - `upper < lower`.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(T lower, T upper, T spacing) noexcept {
    if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(spacing) || spacing <= T(0))
        return 0;

    /// Geometric interval length.
    const T extent = upper - lower;
    if (extent < T(0)) return 0;

    /// Inclusive sample count along the axis.
    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

/**
 * @brief Generate a regular spawn grid and keep only samples accepted by a predicate.
 *
 * @tparam T Floating-point scalar type.
 * @tparam GeometryOperator Geometry query type supporting at least `bound()`.
 * @tparam Predicate Callable of the form `predicate(query, sample, tolerance) -> bool`.
 *
 * @param[out] particles Output device buffer containing accepted sample positions.
 * @param query Geometry query object used both for bounds extraction and predicate testing.
 * @param spacing Cartesian grid spacing.
 * @param tolerance Tolerance forwarded to the predicate.
 * @param predicate User-provided acceptance test for candidate samples.
 *
 * @details
 * The algorithm works in four stages:
 * 1. Compute the axis-aligned bounding box of @p query.
 * 2. Generate a dense Cartesian sampling grid over that bounding box.
 * 3. Evaluate @p predicate for every sample and build a keep mask.
 * 4. Compact accepted samples into the final output buffer using an exclusive scan.
 *
 * The generated grid is inclusive on each axis, i.e. both lower and upper bounds
 * may contribute samples depending on the spacing and floor-based axis count.
 *
 * When no valid samples are accepted, @p particles is cleared.
 *
 * @note
 * The predicate is executed on the device execution path, so it must be compatible
 * with the configured backend.
 */
template <typename T, typename GeometryOperator, typename Predicate>
ATLAS_HOST void
sample_spawn_grid(DeviceBuffer<Vector3<T>>& particles,
                  const GeometryOperator& query,
                  T spacing,
                  T tolerance,
                  Predicate predicate) {
    /// Reset output before generating new samples.
    particles.clear();

    /// Reject invalid spacing immediately.
    if (!std::isfinite(spacing) || spacing <= T(0)) return;

    /// Query the axis-aligned bounds used to build the Cartesian candidate grid.
    const auto bounds = query.bound();

    /// Lower corner of the sampling box.
    const auto& lower = bounds.lower_corner;

    /// Upper corner of the sampling box.
    const auto& upper = bounds.upper_corner;

    /// Number of grid points along x.
    const int nx = sample_axis_count(lower.x, upper.x, spacing);

    /// Number of grid points along y.
    const int ny = sample_axis_count(lower.y, upper.y, spacing);

    /// Number of grid points along z.
    const int nz = sample_axis_count(lower.z, upper.z, spacing);

    /// Abort if any axis cannot produce a valid sample count.
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    /// Total number of candidate grid points.
    const std::size_t total_particles = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    /// Temporary storage for all candidate samples before compaction.
    particles.resize(total_particles);

    /// Keep-mask: 1 for accepted samples, 0 for rejected samples.
    DeviceBuffer<int> keep_mask(total_particles, 0);

    /// Raw pointer to candidate particle positions.
    auto* sampled_particles_ptr = atlas::raw_pointer_cast(particles.data());

    /// Raw pointer to the keep-mask buffer.
    auto* keep_mask_ptr = atlas::raw_pointer_cast(keep_mask.data());

    /**
     * @brief Enumerate every Cartesian grid point and evaluate the acceptance predicate.
     *
     * @details
     * A linear index is mapped into `(ix, iy, iz)` grid coordinates, converted
     * into world-space sample coordinates, written into the candidate buffer,
     * and classified into the keep mask.
     */
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        total_particles,
        [&](std::size_t linear_index) {
            /// Grid x-index derived from the flattened index.
            const int ix = static_cast<int>(linear_index % static_cast<std::size_t>(nx));

            /// Grid y-index derived from the flattened index.
            const int iy = static_cast<int>((linear_index / static_cast<std::size_t>(nx))
                                            % static_cast<std::size_t>(ny));

            /// Grid z-index derived from the flattened index.
            const int iz = static_cast<int>(linear_index
                                            / (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)));

            /// Cartesian sample position at the current grid point.
            const Vector3<T> sample(
                lower.x + static_cast<T>(ix) * spacing,
                lower.y + static_cast<T>(iy) * spacing,
                lower.z + static_cast<T>(iz) * spacing);

            /// Store the candidate sample.
            sampled_particles_ptr[linear_index] = sample;

            /// Record whether the candidate passes the user-supplied filter.
            keep_mask_ptr[linear_index] = predicate(query, sample, tolerance) ? 1 : 0;
        });

    /// Exclusive prefix offsets used to compact accepted samples.
    DeviceBuffer<std::size_t> accepted_offsets(total_particles);

    /**
     * @brief Compute exclusive prefix sums over the keep mask.
     *
     * @details
     * After this scan:
     * - `accepted_offsets[i]` gives the insertion position for sample `i`
     *   if `keep_mask[i] != 0`
     * - the final accepted count is derived from the last scanned offset plus
     *   the final keep-mask value
     */
    atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        keep_mask.begin(),
        keep_mask.end(),
        accepted_offsets.begin(),
        static_cast<std::size_t>(0));

    /// Keep value of the final candidate element.
    int last_keep = 0;

    /// Exclusive offset of the final candidate element.
    std::size_t accepted_count = 0;

    /// Copy the final keep value back to host.
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(keep_mask.data()) + (total_particles - 1),
        &last_keep,
        1);

    /// Copy the final exclusive offset back to host.
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(accepted_offsets.data()) + (total_particles - 1),
        &accepted_count,
        1);

    /// Convert the last exclusive offset into the total number of accepted samples.
    accepted_count += static_cast<std::size_t>(last_keep);

    /// If nothing was accepted, return an empty output.
    if (accepted_count == 0) {
        particles.clear();
        return;
    }

    /// Compact destination buffer containing only accepted samples.
    DeviceBuffer<Vector3<T>> accepted_particles(accepted_count);

    /// Raw pointer to compacted output.
    auto* accepted_particles_ptr = atlas::raw_pointer_cast(accepted_particles.data());

    /// Raw pointer to the exclusive-scan offsets.
    const auto* accepted_offsets_ptr = atlas::raw_pointer_cast(accepted_offsets.data());

    /**
     * @brief Scatter accepted candidates into the compact output buffer.
     *
     * @details
     * Every kept candidate writes itself into the position assigned by the
     * exclusive scan.
     */
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        total_particles,
        [=](std::size_t linear_index) {
            if (keep_mask_ptr[linear_index] != 0) {
                accepted_particles_ptr[accepted_offsets_ptr[linear_index]] = sampled_particles_ptr[linear_index];
            }
        });

    /// Replace the temporary full-grid buffer with the compact accepted sample set.
    particles = std::move(accepted_particles);
}

} // namespace atlas::sampling