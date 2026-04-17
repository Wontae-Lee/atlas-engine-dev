#pragma once

/**
 * @file generate_operator.h
 * @brief Declares a tagged generator operator capable of dispatching among multiple particle generation models.
 */

#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <type_traits>

namespace atlas::fluid {

/**
 * @brief Runtime tag identifying the active particle generation model.
 */
enum class GenerateType : int {
    /**
     * @brief Maxwell-sigma-based generator.
     */
    maxwell_sigma,

    /**
     * @brief Maxwell-Boltzmann-based generator.
     */
    maxwell_boltzmann,

    /**
     * @brief Uniform generator.
     */
    uniform
};

/**
 * @brief Tagged generator operator for particle-attribute generation.
 *
 * This type stores exactly one active generator implementation at a time and
 * dispatches generation requests according to the current runtime tag stored in
 * @ref type.
 *
 * Internally, the object uses a union of concrete generator types:
 * - UniformGenerateOperator<T>
 * - MaxwellSigmaGenerateOperator<T>
 * - MaxwellBoltzmannGenerateOperator<T>
 *
 * Because the active union member is selected dynamically, this type manages
 * construction, destruction, and copying of the active generator manually.
 *
 * @tparam T Floating-point scalar type used by the underlying generators.
 */
template <typename T>
struct GenerateOperator final {
    static_assert(std::is_floating_point_v<T>, "GenerateOperator requires a floating-point T");

    /**
     * @brief Runtime tag indicating which generator is currently active.
     */
    GenerateType type = GenerateType::uniform;

    /**
     * @brief Storage for the active concrete generator.
     *
     * Exactly one member is active at a time, as indicated by @ref type.
     */
    union {
        /**
         * @brief Uniform generator storage.
         */
        UniformGenerateOperator<T> uniform;

        /**
         * @brief Maxwell-sigma generator storage.
         */
        MaxwellSigmaGenerateOperator<T> maxwell_sigma;

        /**
         * @brief Maxwell-Boltzmann generator storage.
         */
        MaxwellBoltzmannGenerateOperator<T> maxwell_boltzmann;
    };

    /**
     * @brief Default constructor.
     *
     * Initializes the operator with a default-constructed uniform generator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator() noexcept;

    /**
     * @brief Constructs a generator operator of the requested type.
     *
     * The selected concrete generator is initialized using the provided seed.
     *
     * @param type Runtime generator type to activate.
     * @param seed Seed used to initialize the concrete generator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(GenerateType type, unsigned int seed = 0u) noexcept;

    /**
     * @brief Copy constructor.
     *
     * Copies the runtime tag and reconstructs the corresponding active generator.
     *
     * @param other Source operator to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(const GenerateOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * Replaces the currently active generator with a copy of the active
     * generator stored in @p other.
     *
     * @param other Source operator to copy from.
     * @return Reference to this object.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator&
    operator=(const GenerateOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * Destroys the currently active concrete generator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~GenerateOperator() noexcept;

    /**
     * @brief Constructs the operator from a uniform generator.
     *
     * The runtime tag is set to @ref GenerateType::uniform.
     *
     * @param op Concrete uniform generator.
     */
    ATLAS_HOST
    GenerateOperator(const UniformGenerateOperator<T>& op);

    /**
     * @brief Constructs the operator from a Maxwell-sigma generator.
     *
     * The runtime tag is set to @ref GenerateType::maxwell_sigma.
     *
     * @param op Concrete Maxwell-sigma generator.
     */
    ATLAS_HOST
    GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op);

    /**
     * @brief Constructs the operator from a Maxwell-Boltzmann generator.
     *
     * The runtime tag is set to @ref GenerateType::maxwell_boltzmann.
     *
     * @param op Concrete Maxwell-Boltzmann generator.
     */
    ATLAS_HOST
    GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op);

    /**
     * @brief Generates a 3D sample using the active generator.
     *
     * The interpretation of @p param0 and @p param1 depends on the active
     * concrete generator type.
     *
     * @param param0 Primary generation parameter.
     * @param param1 Secondary generation parameter. Defaults to 1.
     * @return Generated 3D vector sample.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

private:
    /**
     * @brief Destroys the currently active concrete generator.
     *
     * This function dispatches destruction according to the runtime tag stored
     * in @ref type.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Reconstructs the active generator from another operator.
     *
     * This function assumes that @ref type has already been set to the desired
     * active tag for this object before it is called.
     *
     * @param other Source operator providing the active generator to copy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::GenerateOperator.
 *
 * @tparam T Floating-point scalar type used by the underlying generators.
 */
template <typename T>
using GenerateOperator = atlas::fluid::GenerateOperator<T>;

/**
 * @brief Alias for atlas::fluid::GenerateType.
 */
using GenerateType = atlas::fluid::GenerateType;

} // namespace atlas

#include <atlas/generator/generate_operator.hpp>