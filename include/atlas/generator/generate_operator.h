#pragma once

/**
 * @file generate_operator.h
 * @brief Declares backend-portable particle generation operators.
 *
 * GenerateOperator is the device-friendly counterpart to host-side Generator
 * objects. It erases the concrete distribution choice into a tagged union so
 * source emission code can sample velocities inside backend kernels.
 */

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Identifies the concrete generation law stored in GenerateOperator.
 */
enum class GenerateType : int {
    maxwell_sigma,
    maxwell_boltzmann,
    uniform
};

template <typename T>
struct UniformGenerateOperator final {
    unsigned int seed = 0u;
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit UniformGenerateOperator(unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T min_value,
             T max_value) const;
};

template <typename T>
struct MaxwellSigmaGenerateOperator final {
    unsigned int seed = 0u;
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerateOperator(unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T sigma) const;
};

template <typename T>
struct MaxwellBoltzmannGenerateOperator final {
    unsigned int seed = 0u;
    Vector3<T> bulk_velocity { T(0), T(0), T(0) };
    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellBoltzmannGenerateOperator(
        unsigned int seed               = 0u,
        const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0))) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T temperature,
             T molecular_mass) const;
};

/**
 * @brief Tagged-union wrapper over all supported generation operators.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
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

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::GenerateOperator.
 */
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
