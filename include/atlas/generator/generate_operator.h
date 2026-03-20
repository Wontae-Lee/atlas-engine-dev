#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Runtime velocity-generation mode used by @ref GenerateOperator.
 *
 * @tparam T Floating-point scalar type.
 */
enum class GenerateType : int {
    maxwell_sigma,
    maxwell_boltzmann,
    uniform
};

/**
 * @brief Generates vectors whose components follow a uniform distribution.
 *
 * @details
 * Each component is sampled independently from a uniform distribution over
 * `[min_value, max_value)`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct UniformGenerateOperator final {
    ATLAS_HOST void
    generate(DeviceBuffer<Vector3<T>>& values,
             T min_value,
             T max_value,
             unsigned int seed = 0u) const;
};

/**
 * @brief Generates vectors from independent zero-mean Gaussian velocity components.
 *
 * @details
 * Each Cartesian component is sampled independently from `N(0, sigma^2)`.
 * This is useful when the thermal component standard deviation has already been
 * computed externally.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct MaxwellSigmaGenerateOperator final {
    ATLAS_HOST void
    generate(DeviceBuffer<Vector3<T>>& values,
             T sigma,
             unsigned int seed = 0u) const;
};

/**
 * @brief Generates velocities from the Maxwell-Boltzmann distribution.
 *
 * @details
 * For a gas in thermal equilibrium, each Cartesian velocity component is normally
 * distributed with variance:
 * \f[
 *   \sigma^2 = \frac{k_B T}{m}
 * \f]
 * where:
 * - \f$k_B\f$ is the Boltzmann constant,
 * - \f$T\f$ is the temperature,
 * - \f$m\f$ is the molecular mass.
 *
 * The generated velocity is the sum of:
 * - a thermal component drawn from the Maxwell-Boltzmann distribution
 * - an optional bulk/drift velocity
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct MaxwellBoltzmannGenerateOperator final {
    ATLAS_HOST void
    generate(DeviceBuffer<Vector3<T>>& values,
             T temperature,
             T molecular_mass,
             const Vector3<T>& bulk_velocity,
             unsigned int seed = 0u) const;
};

/**
 * @brief Runtime-dispatched vector generator for uniform or Maxwell modes.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct GenerateOperator final {
    static_assert(std::is_floating_point_v<T>, "GenerateOperator requires a floating-point T");

    GenerateType type = GenerateType::uniform;

    ATLAS_ALL_DEVICE
    explicit GenerateOperator(GenerateType type = GenerateType::uniform) noexcept
        : type(type) { }

    /**
     * @brief Generate vectors using the configured mode.
     *
     * @details
     * Parameter interpretation depends on @ref type:
     * - `GenerateType::uniform`: `param0 = min_value`, `param1 = max_value`
     * - `GenerateType::maxwell_sigma`: `param0 = sigma`, `param1` is ignored
     * - `GenerateType::maxwell_boltzmann`: `param0 = temperature`, `param1 = molecular_mass`
     *
     * @param values Output buffer to overwrite.
     * @param param0 Mode-specific first parameter.
     * @param param1 Mode-specific second parameter.
     * @param seed Seed forwarded to the random engine.
     */
    ATLAS_HOST void
    generate(DeviceBuffer<Vector3<T>>& values,
             T param0,
             T param1 = T(1),
             unsigned int seed = 0u) const;

    /**
     * @brief Generate vectors using the configured mode with an explicit drift velocity.
     *
     * @details
     * For `GenerateType::maxwell_boltzmann`, `bulk_velocity` is added to each sampled
     * thermal velocity after drawing the Maxwell-Boltzmann component values.
     *
     * For `GenerateType::uniform` and `GenerateType::maxwell_sigma`,
     * `bulk_velocity` is ignored.
     *
     * @param values Output buffer to overwrite.
     * @param param0 Mode-specific first parameter.
     * @param param1 Mode-specific second parameter.
     * @param bulk_velocity Explicit drift velocity for Maxwell generation.
     * @param seed Seed forwarded to the random engine.
     */
    ATLAS_HOST void
    generate(DeviceBuffer<Vector3<T>>& values,
             T param0,
             T param1,
             const Vector3<T>& bulk_velocity,
             unsigned int seed = 0u) const;
};

} // namespace atlas::system

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

} // namespace atlas

#include <atlas/generator/generate_operator.hpp>
