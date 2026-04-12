#pragma once

#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas::system {

template <typename T>
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) { }

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {

    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    // Sample each component independently from the same interval.
    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

template <typename T>
MaxwellSigmaGenerateOperator<T>::MaxwellSigmaGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) { }

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const T sigma) const {

    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Draw three independent standard-normal components and scale them by sigma.
    return Vector3<T>(
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine));
}

template <typename T>
MaxwellBoltzmannGenerateOperator<T>::MaxwellBoltzmannGenerateOperator(
    const unsigned int seed,
    const Vector3<T>& bulk_velocity) noexcept
    : seed(seed)
    , bulk_velocity(bulk_velocity)
    , engine(seed) { }

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerateOperator<T>::generate(const T temperature,
                                              const T molecular_mass) const {

    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Thermal speed variance follows sigma^2 = k_B T / m.
    const T sigma = std::sqrt(static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);
    return Vector3<T>(
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine))
        + this->bulk_velocity;
}

template <typename T>
GenerateOperator<T>::GenerateOperator() noexcept
    : type(GenerateType::uniform) {

    // Default to a simple uniform generator so the union is always initialized.
    new (&uniform) UniformGenerateOperator<T> {};
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept
    : type(type) {

    switch (type) {
    case GenerateType::uniform:
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    case GenerateType::maxwell_sigma:
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(seed);
        return;
    case GenerateType::maxwell_boltzmann:
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(seed);
        return;
    default:
        this->type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {
    if (this == &other) return *this;

    // Rebuild the active union member when the type changes.
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
GenerateOperator<T>::~GenerateOperator() noexcept {
    destroy_active();
}

template <typename T>
void
GenerateOperator<T>::destroy_active() noexcept {

    // Destroy only the currently active union member.
    switch (type) {
    case GenerateType::uniform:
        uniform.~UniformGenerateOperator<T>();
        return;
    case GenerateType::maxwell_sigma:
        maxwell_sigma.~MaxwellSigmaGenerateOperator<T>();
        return;
    case GenerateType::maxwell_boltzmann:
        maxwell_boltzmann.~MaxwellBoltzmannGenerateOperator<T>();
        return;
    default:
        uniform.~UniformGenerateOperator<T>();
        return;
    }
}

template <typename T>
void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {

    // Copy-construct the currently active union member.
    switch (type) {
    case GenerateType::uniform:
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    case GenerateType::maxwell_sigma:
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(other.maxwell_sigma);
        return;
    case GenerateType::maxwell_boltzmann:
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(other.maxwell_boltzmann);
        return;
    default:
        type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op)
    : type(GenerateType::uniform) {
    new (&uniform) UniformGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op)
    : type(GenerateType::maxwell_sigma) {
    new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op)
    : type(GenerateType::maxwell_boltzmann) {
    new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(op);
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {

    // Interpret param0/param1 according to the active generation law.
    switch (type) {
    case GenerateType::uniform:
        return uniform.generate(param0, param1);
    case GenerateType::maxwell_sigma:
        return maxwell_sigma.generate(param0);
    case GenerateType::maxwell_boltzmann:
        return maxwell_boltzmann.generate(param0, param1);
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

}
