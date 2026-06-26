#pragma once

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>

namespace atlas {

template <typename T>
struct UniformGenerateOperator final {
    unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED);
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit UniformGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T min_value,
             T max_value) const;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(unsigned int seed,
             T min_value,
             T max_value) const;
};

template <typename T>
struct JitteringGenerateOperator final {
    unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED);
    T base_value { T(0) };
    T jitter_radius { T(0) };
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit JitteringGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED),
        T base_value      = T(0),
        T jitter_radius   = T(0)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1) const;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(unsigned int seed,
             T param0,
             T param1) const;
};

template <typename T>
struct MaxwellSigmaGenerateOperator final {
    unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED);
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T sigma) const;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(unsigned int seed,
             T sigma) const;
};

template <typename T>
struct MaxwellBoltzmannGenerateOperator final {
    unsigned int seed = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED);
    Vector3<T> bulk_velocity { T(0), T(0), T(0) };
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellBoltzmannGenerateOperator(
        unsigned int seed               = static_cast<unsigned int>(atlas::DEFAULT_UNSIGNED_INT_SEED),
        const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0))) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T temperature,
             T molecular_mass) const;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(unsigned int seed,
             T temperature,
             T molecular_mass) const;
};

}

#include <atlas/generator/generate_payload.hpp>