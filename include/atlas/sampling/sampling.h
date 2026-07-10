#pragma once

#include <algorithm>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace atlas {

/**
 * @brief Hash an integer index and seed into a well-mixed 64-bit key.
 *
 * Applies the SplitMix64 finalizer: the index is offset by the seed and a golden-
 * ratio constant, then run through two multiply/xor-shift avalanche rounds and a
 * final xor-shift. The result has good bit-diffusion, so two adjacent indices map
 * to keys with roughly half their bits flipped. This is the stateless primitive
 * behind every hashed-draw helper below; it lets the device generate reproducible
 * randomness from a `(index, seed)` pair without carrying an engine object.
 *
 * @param index Per-item index (e.g. particle slot or cell id); safely reinterpreted
 *              as unsigned, so negative values simply wrap.
 * @param seed  Per-stream seed, typically a time-step value plus a purpose salt.
 * @return A 64-bit hash with avalanche-quality bit mixing.
 * @note Callable from host and device; branch-free and allocation-free.
 * @see atlas::sample_hashed_unit_interval, atlas::sample_hashed_index
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::uint64_t
shuffle_key(const int index, const std::uint64_t seed) noexcept {
    std::uint64_t value = static_cast<std::uint64_t>(index) + seed + atlas::SHUFFLE_HASH_INDEX_OFFSET;
    value               = (value ^ (value >> atlas::SHUFFLE_HASH_FIRST_SHIFT)) * atlas::SHUFFLE_HASH_FIRST_MULTIPLIER;
    value               = (value ^ (value >> atlas::SHUFFLE_HASH_SECOND_SHIFT)) * atlas::SHUFFLE_HASH_SECOND_MULTIPLIER;
    return value ^ (value >> atlas::SHUFFLE_HASH_FINAL_SHIFT);
}

/**
 * @brief Draw two independent standard-normal variates via the Box-Muller transform.
 *
 * Consumes two uniform variates `u1, u2 ∈ [0, 1)` from @p engine and maps them to a
 * pair of independent `N(0, 1)` samples. Producing two at once is the natural output
 * of Box-Muller and lets callers amortize the two uniform draws.
 *
 * @param engine RNG advanced by two draws.
 * @param[out] first  First standard-normal sample (`r·cos θ`).
 * @param[out] second Second standard-normal sample (`r·sin θ`).
 * @note `u1` is clamped up to `eps` before `log` so that a drawn zero cannot yield
 *       `log(0) = -inf`; the radius then uses @ref atlas::sqrt_nonnegative as a
 *       further guard against a negative argument from rounding.
 * @see atlas::generate_standard_normal, atlas::sample_normal_vector
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
generate_standard_normal_pair(atlas::default_random_engine& engine,
                              float& first,
                              float& second) {
    atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float sampled_u1 = dist(engine);
    const float u1         = sampled_u1 > eps ? sampled_u1 : eps;

    const float u2 = dist(engine);

    const float r = atlas::sqrt_nonnegative(-2.0f * std::log(u1));

    const float theta = 2.0f * atlas::pi * u2;

    first  = r * std::cos(theta);
    second = r * std::sin(theta);
}

/**
 * @brief Draw a single standard-normal variate.
 *
 * Convenience wrapper over @ref generate_standard_normal_pair that returns only the
 * first of the pair and discards the second.
 *
 * @param engine RNG advanced by two draws (both Box-Muller uniforms are still
 *               consumed even though one normal is thrown away).
 * @return One sample from `N(0, 1)`.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
generate_standard_normal(atlas::default_random_engine& engine) {
    float first {};
    float second {};
    generate_standard_normal_pair(engine, first, second);
    return first;
}

/**
 * @brief Sample a vector whose components are each uniform on `[min, max)`.
 *
 * Draws the three components independently from the same interval, giving a point
 * uniformly distributed inside the axis-aligned box `[min, max)^3`. Used by the
 * uniform and jittering generators to scatter positions/velocities within a cell.
 *
 * @param engine    RNG advanced by three draws.
 * @param min_value Lower bound applied to every component.
 * @param max_value Upper bound applied to every component (must be `>= min_value`
 *                  for a valid distribution).
 * @return A `Float3` with each component in `[min_value, max_value)`.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_uniform_vector(atlas::default_random_engine& engine,
                      const float min_value,
                      const float max_value) {
    atlas::uniform_real_distribution<float> distribution(min_value, max_value);
    return Float3(
        distribution(engine),
        distribution(engine),
        distribution(engine));
}

/**
 * @brief Sample an isotropic Gaussian velocity with per-axis standard deviation @p sigma.
 *
 * Produces a `Float3` whose three components are independent `N(0, sigma^2)` draws,
 * i.e. a sample from an isotropic 3D Maxwell-Boltzmann-style velocity distribution.
 * Two calls to @ref generate_standard_normal_pair supply the three normals (the
 * fourth is discarded), so four uniform draws are consumed in total.
 *
 * @param engine RNG advanced by four uniform draws.
 * @param sigma  Per-component standard deviation (e.g. the thermal velocity scale).
 * @return A velocity vector drawn from the isotropic Gaussian.
 * @see atlas::generate_standard_normal_pair
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_normal_vector(atlas::default_random_engine& engine,
                     const float sigma) {
    float x {};
    float y {};
    float z {};
    float unused {};
    atlas::generate_standard_normal_pair(engine, x, y);
    atlas::generate_standard_normal_pair(engine, z, unused);
    return Float3(
        sigma * x,
        sigma * y,
        sigma * z);
}

/**
 * @brief Build a right-handed orthonormal basis around a normal, with a safe fallback.
 *
 * Delegates to @ref atlas::orthonormal_basis to produce two unit tangents @p t and
 * @p b orthogonal to @p n. If that fails (typically because @p n is degenerate or
 * near-zero and no stable tangent can be found), the world X/Y axes are substituted
 * so the caller always receives usable, finite vectors rather than NaNs.
 *
 * @param n Reference normal (expected roughly unit-length).
 * @param[out] t First tangent of the basis.
 * @param[out] b Second tangent (bitangent) of the basis.
 * @note The fallback pair is not guaranteed orthogonal to a degenerate @p n; it is a
 *       correctness safety net, not an accuracy guarantee.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
build_orthonormal_basis(const Float3& n,
                        Float3& t,
                        Float3& b) {
    if (!atlas::orthonormal_basis(n, t, b)) {
        t = Float3(1.0f, 0.0f, 0.0f);
        b = Float3(0.0f, 1.0f, 0.0f);
    }
}

/**
 * @brief Sample a direction uniformly over the hemisphere about @p n (solid-angle uniform).
 *
 * Maps two supplied uniform variates to a direction on the hemisphere whose pole is
 * @p n: `cos θ = 1 - u1` is uniform on `[0, 1]` (constant density per unit solid
 * angle) and the azimuth `φ = 2π·u2` is uniform. The direction is reconstructed in
 * the tangent frame of @p n by @ref atlas::spherical_direction.
 *
 * @param n  Hemisphere pole / surface normal (expected unit-length).
 * @param u1 Uniform variate in `[0, 1)` controlling the polar angle.
 * @param u2 Uniform variate in `[0, 1)` controlling the azimuth.
 * @return A unit direction in the hemisphere around @p n.
 * @note The variates are passed in (not drawn here) so the caller can source them
 *       from either an engine or a stateless hash.
 * @warning The `(u1, u2)` roles are the reverse of @ref sample_cosine_hemisphere,
 *          where `u1` drives the azimuth and `u2` the polar angle; keep them
 *          straight when switching between the two.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_uniform_hemisphere(const Float3& n, const float u1, const float u2) {

    const float phi       = 2.0f * atlas::pi * u2;
    const float cos_theta = 1.0f - u1;

    return atlas::spherical_direction(n, cos_theta, phi);
}

/**
 * @brief Sample a direction over the hemisphere about @p n with cosine-weighted density.
 *
 * Produces a direction whose probability density is proportional to `cos θ`, the
 * distribution wanted for diffuse (Lambertian) emission/reflection. Here
 * `cos θ = sqrt(1 - u2)` biases samples toward the pole and `φ = 2π·u1` is the
 * uniform azimuth; @ref atlas::spherical_direction lifts the angles into @p n's frame.
 *
 * @param n  Hemisphere pole / surface normal (expected unit-length).
 * @param u1 Uniform variate in `[0, 1)` controlling the azimuth.
 * @param u2 Uniform variate in `[0, 1)` controlling the polar angle.
 * @return A unit direction, cosine-weighted about @p n.
 * @note @ref atlas::sqrt_nonnegative clamps a slightly-negative `1 - u2` from
 *       rounding to zero rather than yielding NaN.
 * @warning The `(u1, u2)` roles are the reverse of @ref sample_uniform_hemisphere.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_cosine_hemisphere(const Float3& n, const float u1, const float u2) {

    const float phi       = 2.0f * atlas::pi * u1;
    const float cos_theta = atlas::sqrt_nonnegative(1.0f - u2);

    return atlas::spherical_direction(n, cos_theta, phi);
}

/**
 * @brief Sample a unit vector uniformly over the whole sphere.
 *
 * Draws `cos θ = 2·u1 - 1` uniform on `[-1, 1)` and azimuth `φ = 2π·u2`, the standard
 * area-preserving parameterization that spreads points evenly over the unit sphere
 * (no clustering at the poles). Uses the axis-free overload of
 * @ref atlas::spherical_direction, so the result is expressed in world axes.
 *
 * @param engine RNG advanced by two draws.
 * @return A uniformly distributed unit direction.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_random_unit_vector(atlas::default_random_engine& engine) noexcept {
    atlas::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float u1 = dist(engine);
    const float u2 = dist(engine);

    const float cos_theta = 2.0f * u1 - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    return atlas::spherical_direction(cos_theta, phi);
}

/**
 * @brief Sample a unit vector clustered around @p incoming_direction by a sharpness @p alpha.
 *
 * A power-cosine lobe about @p incoming_direction: `cos θ = 2·u1^(1/alpha) - 1`, so
 * larger @p alpha pulls samples toward the axis while `alpha == 1` degenerates to a
 * uniform sphere (`cos θ = 2·u1 - 1`). The azimuth `φ = 2π·u2` is uniform and the
 * result is built in the frame of @p incoming_direction via
 * @ref atlas::spherical_direction.
 *
 * @param incoming_direction Axis the lobe is centered on (expected unit-length).
 * @param alpha              Sharpness exponent; a non-positive value is coerced to
 *                           `1.0f` to keep `1/alpha` finite and the lobe well-defined.
 * @param engine             RNG advanced by two draws.
 * @return A unit direction sampled from the lobe.
 * @note @p alpha is taken by value and reassigned locally; the caller's value is
 *       untouched.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
sample_directional_unit_vector(const Float3& incoming_direction,
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

/**
 * @brief Count the evenly spaced sample points that fit along one axis of a range.
 *
 * Given a closed extent `[lower, upper]` and a step @p spacing, returns
 * `floor((upper - lower) / spacing) + 1` — the number of grid points including both
 * the `lower` endpoint and the last point that does not exceed `upper`. Used by the
 * volume and surface sources to lay out a regular lattice of emission seeds.
 *
 * @param lower   Inclusive lower bound of the axis.
 * @param upper   Inclusive upper bound of the axis.
 * @param spacing Distance between adjacent points; must be strictly positive.
 * @return The point count, or `0` when any argument is non-finite, when @p spacing is
 *         `<= 0`, or when the extent is negative (`upper < lower`).
 * @note A zero-length extent still yields `1` (the single `lower` point), which is the
 *       intended behavior for a degenerate axis.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_axis_count(const float lower, const float upper, const float spacing) noexcept {
    if (!atlas::isfinite(lower) || !atlas::isfinite(upper) || !atlas::isfinite(spacing)
        || spacing <= 0.0f)
        return 0;

    const float extent = upper - lower;
    if (extent < 0.0f) return 0;

    return static_cast<int>(std::floor(extent / spacing)) + 1;
}

/**
 * @brief Stateless pseudo-random value in `[0, 1)` from a 3D seed and a salt (sine hash).
 *
 * The GLSL-style hash `fract(sin(dot(seed, k) + salt) · scale)`: it dots @p seed with
 * the fixed phase coefficients, adds @p salt, takes the sine, blows it up by
 * @ref atlas::RANDOM_HASH_VALUE_SCALE, and returns the fractional part. This lets a
 * kernel derive a repeatable "random" number purely from geometry (a position, a
 * relative velocity) with no engine state; the @p salt decorrelates multiple draws
 * that share one seed.
 *
 * @param seed 3D seed (e.g. a position or velocity vector).
 * @param salt Additive phase offset selecting an independent draw from the same seed.
 * @return A value in `[0, 1)`.
 * @warning This is a low-quality hash: it has visible structure and is sensitive to
 *          the magnitude of @p seed. It is used only where correlations are harmless
 *          (surface-scatter jitter), not for statistically critical sampling.
 * @see atlas::RANDOM_HASH_PHASE_COEFF_X
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sample_hashed_unit_interval(const Float3& seed, const float salt) noexcept {

    const float phase = seed.x * atlas::RANDOM_HASH_PHASE_COEFF_X
        + seed.y * atlas::RANDOM_HASH_PHASE_COEFF_Y
        + seed.z * atlas::RANDOM_HASH_PHASE_COEFF_Z
        + salt;

    const float value = std::sin(phase) * atlas::RANDOM_HASH_VALUE_SCALE;

    return value - std::floor(value);
}

/**
 * @brief Stateless pseudo-random value in `[0, 1)` from an integer index and seed.
 *
 * The high-quality counterpart to the sine-hash overload: it runs `(index, seed)`
 * through @ref atlas::shuffle_key, keeps the top 53 bits (a right shift by
 * @ref atlas::RANDOM_HASH_UNIT_INTERVAL_SHIFT), and scales them by
 * @ref atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE (`1/2^53`) to land in `[0, 1)` with
 * full mantissa resolution. Preferred wherever the draw feeds a statistical test,
 * e.g. the DSMC collision-acceptance probability.
 *
 * @param index Per-item index; combined with @p seed by the hash.
 * @param seed  Per-stream seed (time step plus purpose salt).
 * @return A well-distributed value in `[0, 1)`.
 * @see atlas::shuffle_key
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sample_hashed_unit_interval(const int index, const std::uint64_t seed) noexcept {
    const std::uint64_t value = atlas::shuffle_key(index, seed);
    return static_cast<float>(value >> atlas::RANDOM_HASH_UNIT_INTERVAL_SHIFT)
        * atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE;
}

/**
 * @brief Stateless pseudo-random integer in `[0, upper_bound)` from an index and seed.
 *
 * Hashes `(index, seed)` with @ref atlas::shuffle_key and reduces the result modulo
 * @p upper_bound. The DSMC solver uses this to pick collision partners within a cell
 * without an engine per thread.
 *
 * @param index       Per-item index fed to the hash.
 * @param upper_bound Exclusive upper bound (typically a per-cell particle count).
 * @param seed        Per-stream seed plus a partner-selection salt.
 * @return A value in `[0, upper_bound)`, or `0` when `upper_bound <= 0`.
 * @note The modulo reduction introduces a slight bias when `upper_bound` does not
 *       divide `2^64`; negligible for the small partner counts here.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_hashed_index(const int index,
                    const int upper_bound,
                    const std::uint64_t seed) noexcept {
    if (upper_bound <= 0) {
        return 0;
    }

    const std::uint64_t value = atlas::shuffle_key(index, seed);
    return static_cast<int>(value % static_cast<std::uint64_t>(upper_bound));
}

/**
 * @brief Sample an index in `[0, count)` proportionally to a weight table.
 *
 * Draws one uniform `u ∈ [0, 1)` and walks the running cumulative sum of @p weights,
 * returning the first bucket whose cumulative weight reaches `u` (inverse-CDF /
 * roulette selection). The generators use it to pick a particle species from a set
 * of population ratios.
 *
 * @param weights Pointer to @p count weights; expected to be non-negative and to sum
 *                to `1` (normalized probabilities). Not bounds-checked on the device.
 * @param count   Number of weights.
 * @param engine  RNG advanced by one draw.
 * @return The selected index in `[0, count)`, or `0` when `count <= 0`.
 * @note If the weights sum to less than `1`, a `u` past the last cumulative threshold
 *       falls through to the final index (`count - 1`) — a deliberate clamp that keeps
 *       the return in range rather than reporting an error.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
sample_weighted_index(const float* weights,
                      const int count,
                      atlas::default_random_engine& engine) noexcept {
    if (count <= 0) {
        return 0;
    }

    atlas::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    const float u = distribution(engine);

    float cumulative = 0.0f;
    for (int k = 0; k < count; ++k) {
        cumulative += weights[k];
        if (u <= cumulative) {
            return k;
        }
    }

    return count - 1;
}

/**
 * @brief Weighted draw that returns a table value rather than its index.
 *
 * Selects an index with @ref sample_weighted_index and returns the corresponding
 * entry of the parallel @p values table, truncated to `std::size_t`. In the
 * generators @p values holds the integer species ids, so this maps a species-ratio
 * distribution directly to a species id.
 *
 * @param weights Weight table passed through to @ref sample_weighted_index.
 * @param values  Parallel table of payloads, at least @p count long, indexed by the
 *                selected bucket.
 * @param count   Number of entries in both tables.
 * @param engine  RNG advanced by one draw.
 * @return `static_cast<std::size_t>(values[selected])`, or `0` when `count <= 0`.
 * @warning The float payload is truncated toward zero; callers rely on @p values
 *          holding exact non-negative integers (species ids).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
sample_weighted_choice(const float* weights,
                       const float* values,
                       const int count,
                       atlas::default_random_engine& engine) noexcept {
    if (count <= 0) {
        return std::size_t { 0 };
    }

    return static_cast<std::size_t>(values[sample_weighted_index(weights, count, engine)]);
}

}