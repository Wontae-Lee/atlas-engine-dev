#pragma once

#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

namespace atlas {

template <typename T>
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const unsigned int seed,
                                     const T min_value,
                                     const T max_value) const {
    atlas::default_random_engine<T> seeded_engine(seed);
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    return Vector3<T>(
        dist(seeded_engine),
        dist(seeded_engine),
        dist(seeded_engine));
}

template <typename T>
JitteringGenerateOperator<T>::JitteringGenerateOperator(const unsigned int seed,
                                                        const T base_value,
                                                        const T jitter_radius) noexcept
    : seed(seed)
    , base_value(base_value)
    , jitter_radius(jitter_radius)
    , engine(seed) {
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const T,
                                       const T) const {
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;
    atlas::uniform_real_distribution<T> distribution(-radius, radius);

    return Vector3<T>(
        base_value + distribution(engine),
        base_value + distribution(engine),
        base_value + distribution(engine));
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const unsigned int seed,
                                       const T,
                                       const T) const {
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;
    atlas::default_random_engine<T> seeded_engine(seed);
    atlas::uniform_real_distribution<T> distribution(-radius, radius);

    return Vector3<T>(
        base_value + distribution(seeded_engine),
        base_value + distribution(seeded_engine),
        base_value + distribution(seeded_engine));
}

template <typename T>
MaxwellSigmaGenerateOperator<T>::MaxwellSigmaGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const T sigma) const {
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::generate_standard_normal_pair<T>(engine, x, y);
    atlas::generate_standard_normal_pair<T>(engine, z, unused);

    return Vector3<T>(
        sigma * x,
        sigma * y,
        sigma * z);
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const unsigned int seed,
                                          const T sigma) const {
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    atlas::default_random_engine<T> seeded_engine(seed);
    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::generate_standard_normal_pair<T>(seeded_engine, x, y);
    atlas::generate_standard_normal_pair<T>(seeded_engine, z, unused);

    return Vector3<T>(
        sigma * x,
        sigma * y,
        sigma * z);
}

template <typename T>
MaxwellBoltzmannGenerateOperator<T>::MaxwellBoltzmannGenerateOperator(
    const unsigned int seed,
    const Vector3<T>& bulk_velocity) noexcept
    : seed(seed)
    , bulk_velocity(bulk_velocity)
    , engine(seed) {
}

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerateOperator<T>::generate(const T temperature,
                                              const T molecular_mass) const {
    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const T sigma = atlas::sqrt_nonnegative(
        static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);
    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::generate_standard_normal_pair<T>(engine, x, y);
    atlas::generate_standard_normal_pair<T>(engine, z, unused);

    return Vector3<T>(
               sigma * x,
               sigma * y,
               sigma * z)
        + this->bulk_velocity;
}

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerateOperator<T>::generate(const unsigned int seed,
                                              const T temperature,
                                              const T molecular_mass) const {
    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const T sigma = atlas::sqrt_nonnegative(
        static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);
    atlas::default_random_engine<T> seeded_engine(seed);
    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::generate_standard_normal_pair<T>(seeded_engine, x, y);
    atlas::generate_standard_normal_pair<T>(seeded_engine, z, unused);

    return Vector3<T>(
               sigma * x,
               sigma * y,
               sigma * z)
        + this->bulk_velocity;
}

} // namespace atlas
