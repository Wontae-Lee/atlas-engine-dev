#pragma once

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <type_traits>

namespace atlas::system {

enum class GenerateType : int {
    maxwell_sigma,
    maxwell_boltzmann,
    uniform
};

template <typename T>
struct UniformGenerateOperator final {
    unsigned int seed = 0u;
    mutable atlas::default_random_engine<T> engine;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit UniformGenerateOperator(unsigned int seed = 0u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T min_value,
             T max_value) const;
};

template <typename T>
struct MaxwellSigmaGenerateOperator final {
    unsigned int seed = 0u;
    mutable atlas::default_random_engine<T> engine;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerateOperator(unsigned int seed = 0u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T sigma) const;
};

template <typename T>
struct MaxwellBoltzmannGenerateOperator final {
    unsigned int seed = 0u;
    mutable atlas::default_random_engine<T> engine;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit MaxwellBoltzmannGenerateOperator(unsigned int seed = 0u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T temperature,
             T molecular_mass,
             const Vector3<T>& bulk_velocity) const;
};

template <typename T>
struct GenerateOperator final {
    static_assert(std::is_floating_point_v<T>, "GenerateOperator requires a floating-point T");

    GenerateType type = GenerateType::uniform;
    union {
        UniformGenerateOperator<T> uniform;
        MaxwellSigmaGenerateOperator<T> maxwell_sigma;
        MaxwellBoltzmannGenerateOperator<T> maxwell_boltzmann;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(GenerateType type, unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator&
    operator=(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~GenerateOperator() noexcept;

    ATLAS_HOST
    GenerateOperator(const UniformGenerateOperator<T>& op);

    ATLAS_HOST
    GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op);

    ATLAS_HOST
    GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1,
             const Vector3<T>& bulk_velocity) const;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

}

namespace atlas {

template <typename T>
using GenerateOperator = atlas::system::GenerateOperator<T>;

template <typename T>
using UniformGenerateOperator = atlas::system::UniformGenerateOperator<T>;

template <typename T>
using MaxwellSigmaGenerateOperator = atlas::system::MaxwellSigmaGenerateOperator<T>;

template <typename T>
using MaxwellBoltzmannGenerateOperator = atlas::system::MaxwellBoltzmannGenerateOperator<T>;

using GenerateType = atlas::system::GenerateType;

}

#include <atlas/generator/generate_operator.hpp>