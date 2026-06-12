#pragma once

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
    return atlas::sample_uniform_vector<T>(engine, min_value, max_value);
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const unsigned int seed,
                                     const T min_value,
                                     const T max_value) const {
    atlas::default_random_engine<T> seeded_engine(seed);
    return atlas::sample_uniform_vector<T>(seeded_engine, min_value, max_value);
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
    return Vector3<T>(base_value, base_value, base_value)
        + atlas::sample_uniform_vector<T>(engine, -radius, radius);
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const unsigned int seed,
                                       const T,
                                       const T) const {
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;
    atlas::default_random_engine<T> seeded_engine(seed);
    return Vector3<T>(base_value, base_value, base_value)
        + atlas::sample_uniform_vector<T>(seeded_engine, -radius, radius);
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

    return atlas::sample_normal_vector<T>(engine, sigma);
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const unsigned int seed,
                                          const T sigma) const {
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    atlas::default_random_engine<T> seeded_engine(seed);
    return atlas::sample_normal_vector<T>(seeded_engine, sigma);
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
    return atlas::sample_normal_vector<T>(engine, sigma) + this->bulk_velocity;
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
    return atlas::sample_normal_vector<T>(seeded_engine, sigma) + this->bulk_velocity;
}

} // namespace atlas
