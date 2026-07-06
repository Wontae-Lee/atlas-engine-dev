#pragma once

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

/**
 * @file generate_payload.h
 * @brief The four concrete velocity/vector generators
 *        `Source`/`Fluid::generators()` draw new particle velocities
 *        from, each a device-callable value type wrapping a stateful RNG
 *        engine (`atlas::default_random_engine`).
 *
 * @details
 * ### Background — sampling a Maxwellian velocity distribution
 * `MaxwellBoltzmannGenerate` is the physically-motivated
 * generator `Source` normally uses for particle emission: kinetic theory
 * shows that a gas in equilibrium at temperature `T` has each Cartesian
 * velocity component independently normally distributed,
 * `p(v_x) ~ exp(-m v_x^2 / (2 k_B T))`, i.e.
 * `v_x, v_y, v_z ~ Normal(0, sigma)`, `sigma = sqrt(k_B T / m)` — the
 * three-component vector `v` this produces (before any bulk-flow
 * offset) is exactly a Maxwell-Boltzmann-distributed velocity. This is
 * the standard result that the Maxwell-Boltzmann speed distribution
 * factors into three independent 1D Gaussians, one per axis;
 * `MaxwellSigmaGenerate` is the same isotropic-Gaussian sampler
 * with `sigma` supplied directly (for callers that already have it,
 * skipping the `sqrt(k_B T / m)` conversion), and
 * `MaxwellBoltzmannGenerate` additionally adds a configurable
 * `bulk_velocity` drift — `v = v_thermal + u_bulk`, the standard way to
 * superimpose a mean flow on top of thermal (random) motion.
 *
 * `UniformGenerate` and `JitteringGenerate` are
 * non-physical generators for testing/simplified sources:
 * `UniformGenerate` draws each component uniformly in
 * `[min_value, max_value]`; `JitteringGenerate` returns a fixed
 * `base_value` plus small uniform noise in `[-jitter_radius,
 * jitter_radius]` per component, ignoring whatever temperature/mass
 * parameters are passed to `generate()` — useful for a near-deterministic
 * source (e.g. a fixed injection velocity with a small spread) rather
 * than a Maxwellian one.
 *
 * Every operator exposes two `generate()` overloads: one using its own
 * internal `mutable engine` (a stateful stream, advancing across calls —
 * appropriate for host-side, sequential use), and one that constructs a
 * fresh `default_random_engine` from an explicit `seed_` per call
 * (stateless — the pattern `Source` uses, since each
 * device thread emitting one particle needs an independent, reproducible
 * draw without sharing mutable RNG state across threads).
 */

namespace atlas {

/** @brief Draws each velocity component uniformly in `[min_value,
 *  max_value]`; non-physical, for testing/simplified sources. */
struct UniformGenerate final {
    unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
    mutable atlas::default_random_engine engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit UniformGenerate(
        const unsigned int seed_ = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept
        : seed(seed_)
        , engine(seed_) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const float min_value, const float max_value) const {
        return atlas::sample_uniform_vector(engine, min_value, max_value);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const unsigned int seed_, const float min_value, const float max_value) const {
        atlas::default_random_engine seeded_engine(seed_);
        return atlas::sample_uniform_vector(seeded_engine, min_value, max_value);
    }
};

/** @brief Returns `base_value + Uniform(-jitter_radius, jitter_radius)`
 *  per component, ignoring the `generate()` temperature/mass arguments;
 *  a fixed value with small noise rather than a Maxwellian draw. */
struct JitteringGenerate final {
    unsigned int seed   = atlas::DEFAULT_UNSIGNED_INT_SEED;
    float base_value    = 0.0f;
    float jitter_radius = 0.0f;
    mutable atlas::default_random_engine engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit JitteringGenerate(
        const unsigned int seed_   = atlas::DEFAULT_UNSIGNED_INT_SEED,
        const float base_value_    = 0.0f,
        const float jitter_radius_ = 0.0f) noexcept
        : seed(seed_)
        , base_value(base_value_)
        , jitter_radius(jitter_radius_)
        , engine(seed_) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const float, const float) const {
        const float radius = std::abs(jitter_radius);
        return Float3(base_value, base_value, base_value)
            + atlas::sample_uniform_vector(engine, -radius, radius);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const unsigned int seed_, const float, const float) const {
        const float radius = std::abs(jitter_radius);
        atlas::default_random_engine seeded_engine(seed_);
        return Float3(base_value, base_value, base_value)
            + atlas::sample_uniform_vector(seeded_engine, -radius, radius);
    }
};

/** @brief Isotropic 3D Gaussian draw with `sigma` supplied directly
 *  (`Normal(0, sigma)` per component) — the Maxwellian thermal-velocity
 *  shape without the `sqrt(k_B T / m)` conversion; see this file's
 *  top-of-file documentation. Returns zero if `sigma <= 0`. */
struct MaxwellSigmaGenerate final {
    unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
    mutable atlas::default_random_engine engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerate(
        const unsigned int seed_ = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept
        : seed(seed_)
        , engine(seed_) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const float sigma) const {
        if (!(sigma > 0.0f)) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        return atlas::sample_normal_vector(engine, sigma);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const unsigned int seed_, const float sigma) const {
        if (!(sigma > 0.0f)) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        atlas::default_random_engine seeded_engine(seed_);
        return atlas::sample_normal_vector(seeded_engine, sigma);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const float sigma, const float) const {
        return generate(sigma);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const unsigned int seed_, const float sigma, const float) const {
        return generate(seed_, sigma);
    }
};

/**
 * @brief Physically-motivated Maxwellian velocity draw with bulk flow:
 *        `v = Normal(0, sqrt(k_B*T/m)) + bulk_velocity`. See this file's
 *        top-of-file documentation for the equilibrium kinetic-theory
 *        derivation. Returns zero if `temperature`/`molecular_mass` is
 *        non-positive.
 */
struct MaxwellBoltzmannGenerate final {
    unsigned int seed    = atlas::DEFAULT_UNSIGNED_INT_SEED;
    Float3 bulk_velocity = Float3(0.0f, 0.0f, 0.0f);
    mutable atlas::default_random_engine engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellBoltzmannGenerate(
        const unsigned int seed_     = atlas::DEFAULT_UNSIGNED_INT_SEED,
        const Float3& bulk_velocity_ = Float3(0.0f, 0.0f, 0.0f)) noexcept
        : seed(seed_)
        , bulk_velocity(bulk_velocity_)
        , engine(seed_) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const float temperature, const float molecular_mass) const {
        if (!(temperature > 0.0f) || !(molecular_mass > 0.0f)) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const float sigma = atlas::sqrt_nonnegative(
            atlas::boltzmann_constant * temperature / molecular_mass);
        return atlas::sample_normal_vector(engine, sigma) + bulk_velocity;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(const unsigned int seed_,
             const float temperature,
             const float molecular_mass) const {
        if (!(temperature > 0.0f) || !(molecular_mass > 0.0f)) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const float sigma = atlas::sqrt_nonnegative(
            atlas::boltzmann_constant * temperature / molecular_mass);
        atlas::default_random_engine seeded_engine(seed_);
        return atlas::sample_normal_vector(seeded_engine, sigma) + bulk_velocity;
    }
};

}
