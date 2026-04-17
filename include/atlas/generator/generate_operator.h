#pragma once

/**
 * @file generate_operator.h
 * @brief Declares a backend-portable tagged-union wrapper for particle generation laws.
 *
 * @details
 * This header defines @ref atlas::system::GenerateOperator, a lightweight
 * device-friendly wrapper that stores one of the supported particle-generation
 * operators behind a runtime type tag.
 *
 * The purpose of this abstraction is to bridge:
 * - host-side generator objects that may be selected polymorphically or through
 *   builder-style APIs, and
 * - backend/device execution code that requires a compact, trivially passable,
 *   non-virtual representation.
 *
 * ## Design overview
 * Rather than storing generators through inheritance or dynamic allocation,
 * @ref GenerateOperator uses:
 * - a runtime discriminator @ref GenerateType, and
 * - a tagged union containing the concrete operator payload.
 *
 * This allows source-emission or particle-initialization code to:
 * - store different generator laws in a uniform array,
 * - branch on the active generation law inside kernels,
 * - invoke generation logic through a single interface.
 *
 * ## Supported generation laws
 * The operator currently supports:
 * - @ref GenerateType::uniform
 * - @ref GenerateType::maxwell_sigma
 * - @ref GenerateType::maxwell_boltzmann
 *
 * Each case corresponds to a concrete backend-portable generator operator type.
 *
 * ## Lifetime management
 * Since the active payload is stored in a union, @ref GenerateOperator is
 * responsible for:
 * - constructing the correct active union member,
 * - destroying the active union member,
 * - copying the active member according to the current type tag.
 *
 * Those responsibilities are handled internally through helper functions such as
 * @ref destroy_active and @ref copy_from.
 *
 * ## Host/device usage
 * - Constructors from concrete operator objects are host-side convenience paths.
 * - Core storage, copy, destruction, and generation are available in
 *   `ATLAS_ALL_DEVICE` contexts so the wrapper can be used in backend code.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation and generator outputs.
 */

#include <type_traits>

namespace atlas::system {

/**
 * @brief Identifies the concrete generation law stored in @ref GenerateOperator.
 *
 * @details
 * This enum acts as the runtime discriminator for the tagged union stored in
 * @ref GenerateOperator.
 *
 * It is used to:
 * - determine which concrete generator payload is currently active,
 * - dispatch generation logic inside host or device code,
 * - correctly manage union lifetime during copy and destruction.
 */
enum class GenerateType : int {
    maxwell_sigma,     ///< Generator based on a Maxwell distribution parameterized by sigma-like input.
    maxwell_boltzmann, ///< Generator based on a Maxwell-Boltzmann distribution.
    uniform            ///< Generator based on a uniform distribution law.
};

} // namespace atlas::system

#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>

namespace atlas::system {

/**
 * @brief Tagged-union wrapper over all supported particle generation operators.
 *
 * @details
 * @ref GenerateOperator stores exactly one active concrete generator operator
 * together with a runtime type tag describing which union member is currently valid.
 *
 * This type is intended to be:
 * - compact,
 * - backend-portable,
 * - non-polymorphic,
 * - easy to copy into device kernels,
 * - suitable for storage in arrays or buffers of mixed generator laws.
 *
 * ## Internal representation
 * The wrapper contains:
 * - @ref type : runtime discriminator identifying the active generator law,
 * - a union of concrete generator operator payloads.
 *
 * At any given time, exactly one union member is expected to be active and
 * consistent with the current value of @ref type.
 *
 * ## Typical usage
 * A @ref GenerateOperator is commonly created:
 * - by default construction,
 * - from an explicit @ref GenerateType and optional seed,
 * - from one of the concrete operator types,
 * - by copy construction from another @ref GenerateOperator.
 *
 * It can then be used through @ref generate to sample a generated vector value.
 *
 * ## Parameters to generate()
 * The exact interpretation of `param0` and `param1` depends on the active
 * generation law. For example, they may represent quantities such as:
 * - a scale parameter,
 * - a temperature-like quantity,
 * - a mass or variance-related parameter,
 * - a component bound for uniform sampling.
 *
 * The concrete meaning is delegated to the active generator implementation.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct GenerateOperator final {
    static_assert(std::is_floating_point_v<T>, "GenerateOperator requires a floating-point T");

    /**
     * @brief Runtime discriminator describing the active generator payload.
     *
     * @details
     * This value determines which union member is currently active and therefore
     * which generation law is used by @ref generate.
     */
    GenerateType type = GenerateType::uniform;

    /**
     * @brief Tagged union storing the active concrete generator operator.
     *
     * @details
     * Exactly one of the following members is expected to be active at a time,
     * consistent with the current value of @ref type.
     */
    union {
        /**
         * @brief Uniform generation operator payload.
         */
        UniformGenerateOperator<T> uniform;

        /**
         * @brief Maxwell-sigma generation operator payload.
         */
        MaxwellSigmaGenerateOperator<T> maxwell_sigma;

        /**
         * @brief Maxwell-Boltzmann generation operator payload.
         */
        MaxwellBoltzmannGenerateOperator<T> maxwell_boltzmann;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs the wrapper with a default active generator law. By default,
     * the runtime type is initialized to @ref GenerateType::uniform and the
     * matching union member is expected to be constructed accordingly.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator() noexcept;

    /**
     * @brief Construct the wrapper from an explicit generator type and seed.
     *
     * @details
     * Activates the union member corresponding to @p type and initializes it
     * using the provided seed, if applicable to the concrete generator.
     *
     * @param type Runtime generation law to activate.
     * @param seed Optional seed used to initialize the active generator payload.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(GenerateType type, unsigned int seed = 0u) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies the runtime discriminator and constructs the corresponding active
     * union member from @p other.
     *
     * @param other Source operator to copy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(const GenerateOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Destroys the currently active payload, copies the runtime discriminator from
     * @p other, and reconstructs the correct active union member.
     *
     * @param other Source operator to copy from.
     * @return `*this`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator&
    operator=(const GenerateOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * @details
     * Destroys the currently active union member according to the stored
     * runtime discriminator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~GenerateOperator() noexcept;

    /**
     * @brief Construct the wrapper from a uniform generation operator.
     *
     * @details
     * Activates the @ref uniform union member and sets @ref type to
     * @ref GenerateType::uniform.
     *
     * @param op Concrete uniform operator to store.
     */
    ATLAS_HOST
    GenerateOperator(const UniformGenerateOperator<T>& op);

    /**
     * @brief Construct the wrapper from a Maxwell-sigma generation operator.
     *
     * @details
     * Activates the @ref maxwell_sigma union member and sets @ref type to
     * @ref GenerateType::maxwell_sigma.
     *
     * @param op Concrete Maxwell-sigma operator to store.
     */
    ATLAS_HOST
    GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op);

    /**
     * @brief Construct the wrapper from a Maxwell-Boltzmann generation operator.
     *
     * @details
     * Activates the @ref maxwell_boltzmann union member and sets @ref type to
     * @ref GenerateType::maxwell_boltzmann.
     *
     * @param op Concrete Maxwell-Boltzmann operator to store.
     */
    ATLAS_HOST
    GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op);

    /**
     * @brief Generate a vector sample using the active generation law.
     *
     * @details
     * Dispatches to the currently active concrete generator payload according to
     * the value of @ref type and returns the generated vector sample.
     *
     * The semantic meaning of @p param0 and @p param1 depends on the active
     * generator type and is defined by the corresponding concrete operator.
     *
     * @param param0 Primary generator parameter.
     * @param param1 Secondary generator parameter. Defaults to `1`.
     * @return Generated vector sample.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

private:
    /**
     * @brief Destroy the currently active union member.
     *
     * @details
     * Uses the stored runtime discriminator to invoke the correct destructor for
     * the active payload.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Copy the active payload from another wrapper.
     *
     * @details
     * Constructs the correct union member from @p other according to its
     * runtime discriminator.
     *
     * @param other Source wrapper to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::GenerateOperator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GenerateOperator = atlas::system::GenerateOperator<T>;

/**
 * @brief Convenience alias for @ref atlas::system::GenerateType.
 */
using GenerateType = atlas::system::GenerateType;

} // namespace atlas

#include <atlas/generator/generate_operator.hpp>