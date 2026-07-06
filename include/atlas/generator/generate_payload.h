#pragma once

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas {

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
