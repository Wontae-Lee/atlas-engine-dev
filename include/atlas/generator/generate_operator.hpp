#pragma once

#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas::system {

template <typename T>
void
UniformGenerateOperator<T>::generate(DeviceBuffer<Vector3<T>>& values,
                                     const T min_value,
                                     const T max_value,
                                     const unsigned int seed) const {
    if (values.empty()) return;

    atlas::default_random_engine<T> engine(seed);
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    for (auto& value : values) {
        value = Vector3<T>(
            dist(engine),
            dist(engine),
            dist(engine));
    }
}

template <typename T>
void
MaxwellSigmaGenerateOperator<T>::generate(DeviceBuffer<Vector3<T>>& values,
                                          const T sigma,
                                          const unsigned int seed) const {
    if (values.empty()) return;

    atlas::default_random_engine<T> engine(seed);

    for (auto& value : values) {
        value = Vector3<T>(
            sigma * atlas::sampling::generate_standard_normal<T>(engine),
            sigma * atlas::sampling::generate_standard_normal<T>(engine),
            sigma * atlas::sampling::generate_standard_normal<T>(engine));
    }
}

template <typename T>
void
MaxwellBoltzmannGenerateOperator<T>::generate(DeviceBuffer<Vector3<T>>& values,
                                              const T temperature,
                                              const T molecular_mass,
                                              const Vector3<T>& bulk_velocity,
                                              const unsigned int seed) const {
    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        values.clear();
        return;
    }

    const T sigma = std::sqrt(static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);
    MaxwellSigmaGenerateOperator<T> {}.generate(values, sigma, seed);

    for (auto& value : values) {
        value += bulk_velocity;
    }
}

template <typename T>
void
GenerateOperator<T>::generate(DeviceBuffer<Vector3<T>>& values,
                              const T param0,
                              const T param1,
                              const unsigned int seed) const {
    switch (type) {
    case GenerateType::maxwell_sigma:
        MaxwellSigmaGenerateOperator<T> {}.generate(values, param0, seed);
        return;
    case GenerateType::maxwell_boltzmann:
        MaxwellBoltzmannGenerateOperator<T> {}.generate(
            values,
            param0,
            param1,
            Vector3<T>(T(0), T(0), T(0)),
            seed);
        return;
    case GenerateType::uniform:
        UniformGenerateOperator<T> {}.generate(values, param0, param1, seed);
        return;
    }
}

template <typename T>
void
GenerateOperator<T>::generate(DeviceBuffer<Vector3<T>>& values,
                              const T param0,
                              const T param1,
                              const Vector3<T>& bulk_velocity,
                              const unsigned int seed) const {
    switch (type) {
    case GenerateType::maxwell_sigma:
        MaxwellSigmaGenerateOperator<T> {}.generate(values, param0, seed);
        return;
    case GenerateType::maxwell_boltzmann:
        MaxwellBoltzmannGenerateOperator<T> {}.generate(values, param0, param1, bulk_velocity, seed);
        return;
    case GenerateType::uniform:
        UniformGenerateOperator<T> {}.generate(values, param0, param1, seed);
        return;
    }
}

} // namespace atlas::system
