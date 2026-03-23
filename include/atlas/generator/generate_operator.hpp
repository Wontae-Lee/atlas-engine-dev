#pragma once

#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas::system {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) { }

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {
    // The engine is stored inside the operator so repeated generate() calls
    // advance a deterministic sequence for a given seed.
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
MaxwellSigmaGenerateOperator<T>::MaxwellSigmaGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) { }

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const T sigma) const {
    // Degenerate sigma values collapse to zero velocity rather than producing
    // invalid normal-distribution parameters.
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Each Cartesian component is sampled independently from N(0, sigma^2).
    return Vector3<T>(
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
MaxwellBoltzmannGenerateOperator<T>::MaxwellBoltzmannGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) { }

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector3<T>
MaxwellBoltzmannGenerateOperator<T>::generate(const T temperature,
                                              const T molecular_mass,
                                              const Vector3<T>& bulk_velocity) const {
    // Invalid thermodynamic parameters map to a zero vector to keep the
    // runtime operator total and side-effect free.
    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Convert the physical parameters into the equivalent component-wise
    // Gaussian sigma, then add the optional drift velocity.
    const T sigma = std::sqrt(static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);
    return Vector3<T>(
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine))
           + bulk_velocity;
}

/* ====================================================================== */
/* GenerateOperator special members                                        */
/* ====================================================================== */

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
GenerateOperator<T>::GenerateOperator() noexcept
    : type(GenerateType::uniform) {
    // Default-construct the active union member to keep the tagged union valid.
    new (&uniform) UniformGenerateOperator<T> {};
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept
    : type(type) {
    // Manual tagged-union construction keeps the runtime wrapper trivially
    // lightweight while still supporting multiple generator families.
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
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {
    if (this == &other) return *this;
    // Rebuild the active union member because the source and destination tags
    // may refer to different generator types.
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
GenerateOperator<T>::~GenerateOperator() noexcept {
    destroy_active();
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
GenerateOperator<T>::destroy_active() noexcept {
    // Only the member selected by `type` is alive at any point.
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
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {
    // Copy-construct the active union alternative matching the already-copied tag.
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

/* ====================================================================== */
/* GenerateOperator tagged constructors                                    */
/* ====================================================================== */

template <typename T>
ATLAS_HOST
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op)
    : type(GenerateType::uniform) {
    new (&uniform) UniformGenerateOperator<T>(op);
}

template <typename T>
ATLAS_HOST
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op)
    : type(GenerateType::maxwell_sigma) {
    new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(op);
}

template <typename T>
ATLAS_HOST
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op)
    : type(GenerateType::maxwell_boltzmann) {
    new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(op);
}

/* ====================================================================== */
/* Generate dispatch                                                       */
/* ====================================================================== */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {
    // Dispatch to the concrete generator while interpreting the generic
    // parameters according to the active runtime tag.
    switch (type) {
    case GenerateType::uniform:
        return uniform.generate(param0, param1);
    case GenerateType::maxwell_sigma:
        return maxwell_sigma.generate(param0);
    case GenerateType::maxwell_boltzmann:
        return maxwell_boltzmann.generate(
            param0,
            param1,
            Vector3<T>(T(0), T(0), T(0)));
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1,
                              const Vector3<T>& bulk_velocity) const {
    // The 3-argument overload only affects Maxwell-Boltzmann generation; the
    // other modes intentionally ignore the drift term.
    switch (type) {
    case GenerateType::uniform:
        return uniform.generate(param0, param1);
    case GenerateType::maxwell_sigma:
        return maxwell_sigma.generate(param0);
    case GenerateType::maxwell_boltzmann:
        return maxwell_boltzmann.generate(param0, param1, bulk_velocity);
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

} // namespace atlas::system
