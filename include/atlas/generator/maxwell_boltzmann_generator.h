#pragma once

/**
 * @file maxwell_boltzmann_generator.h
 * @brief Declares Maxwell-Boltzmann particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::fluid::MaxwellBoltzmannGenerateOperator, a backend-portable
 *   operator that samples particle velocities according to a Maxwell-Boltzmann
 *   distribution, and
 * - @ref atlas::fluid::MaxwellBoltzmannGenerator, a host-side generator class
 *   that owns the corresponding runtime parameters and can export a portable
 *   @ref GenerateOperator.
 *
 * ## Physical interpretation
 * The Maxwell-Boltzmann distribution is commonly used to model thermal particle
 * velocities for gases in equilibrium or near-equilibrium settings.
 *
 * In this formulation, sampled velocities are influenced by:
 * - a thermal temperature,
 * - a molecular mass,
 * - an optional bulk or drift velocity that shifts the distribution mean.
 *
 * ## Host/device split
 * Atlas separates this generation law into two complementary representations:
 * - a host-side polymorphic generator object for runtime configuration and
 *   direct host sampling,
 * - a compact backend-friendly operator for use inside device kernels or
 *   backend-parallel emission code.
 *
 * ## Construction
 * The host-side generator may be:
 * - constructed directly from its physical parameters, or
 * - built through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for parameters and generated velocities.
 */

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <atlas/generator/generator.h>
#include <optional>

namespace atlas::fluid {

/**
 * @brief Backend-portable operator for sampling Maxwell-Boltzmann-distributed velocities.
 *
 * @details
 * @ref MaxwellBoltzmannGenerateOperator is the lightweight, device-friendly
 * representation of the Maxwell-Boltzmann generation law.
 *
 * It stores:
 * - a deterministic random seed,
 * - an optional bulk velocity used to shift the sampled distribution,
 * - an internal random engine used to produce pseudo-random samples.
 *
 * ## Usage
 * The operator is typically embedded inside a @ref GenerateOperator tagged union
 * and consumed by source-emission code in host or device contexts.
 *
 * The actual sampling is performed through @ref generate, which interprets:
 * - `temperature` as the thermal state controlling the distribution scale,
 * - `molecular_mass` as the mass parameter governing the thermal spread.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct MaxwellBoltzmannGenerateOperator final {
    /**
     * @brief Deterministic seed used to initialize the random engine.
     *
     * @details
     * This seed controls the reproducibility of the generated velocity samples.
     */
    unsigned int seed = atlas::seed::DEFAULT_UNSIGNED_INT_SEED;

    /**
     * @brief Mean drift velocity added to the sampled thermal fluctuation.
     *
     * @details
     * This vector shifts the sampled distribution so the generated velocities are
     * centered around a moving bulk frame rather than around zero.
     */
    Vector3<T> bulk_velocity { T(0), T(0), T(0) };

    /**
     * @brief Internal pseudo-random engine used for sampling.
     *
     * @details
     * The exact random-number generation behavior is defined by
     * @ref atlas::default_random_engine.
     */
    mutable atlas::default_random_engine<T> engine;

    /**
     * @brief Construct a Maxwell-Boltzmann generation operator.
     *
     * @details
     * Initializes the random seed, bulk velocity, and internal random engine.
     *
     * @param seed Deterministic seed for reproducible sampling.
     * @param bulk_velocity Mean drift velocity of the generated distribution.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellBoltzmannGenerateOperator(
        unsigned int seed               = atlas::seed::DEFAULT_UNSIGNED_INT_SEED,
        const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0))) noexcept;

    /**
     * @brief Generate a velocity sample from a Maxwell-Boltzmann distribution.
     *
     * @details
     * Produces a thermal velocity sample parameterized by:
     * - the supplied temperature,
     * - the supplied molecular mass,
     * - the stored bulk velocity offset.
     *
     * The exact numerical sampling method is implementation-defined in
     * `maxwell_boltzmann_generator.hpp`.
     *
     * @param temperature Thermal temperature controlling the distribution width.
     * @param molecular_mass Molecular mass controlling the thermal spread.
     * @return Generated particle velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T temperature,
             T molecular_mass) const;
};

/**
 * @brief Host-side generator for Maxwell-Boltzmann-distributed velocities.
 *
 * @details
 * @ref MaxwellBoltzmannGenerator is the polymorphic host-side representation of
 * the Maxwell-Boltzmann particle-velocity generation law.
 *
 * It stores the physical parameters needed to generate samples:
 * - temperature,
 * - molecular mass,
 * - bulk velocity,
 * - random seed.
 *
 * It also caches a backend-portable @ref GenerateOperator so runtime fluids can
 * export the generation law for device-side emission without reconstructing it
 * repeatedly.
 *
 * ## Responsibilities
 * A generator instance can:
 * - sample velocities directly on the host via @ref generate,
 * - report its scalar parameters through @ref param0 and @ref param1,
 * - expose its runtime type through @ref type,
 * - return or create a backend-portable operator representation.
 *
 * ## Parameter mapping
 * For this generator:
 * - @ref param0 corresponds to temperature,
 * - @ref param1 corresponds to molecular mass.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class MaxwellBoltzmannGenerator final : public Generator<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing
     *        @ref MaxwellBoltzmannGenerator.
     *
     * @details
     * The builder stages the physical parameters of the generator, validates
     * them, and then constructs either:
     * - a generator by value, or
     * - a host-owned shared pointer to a generator.
     */
    class Builder;

public:
    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Construct a Maxwell-Boltzmann generator from explicit parameters.
     *
     * @param temperature Thermal temperature.
     * @param molecular_mass Molecular mass.
     * @param bulk_velocity Mean drift velocity added to sampled thermal motion.
     * @param seed Deterministic random seed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellBoltzmannGenerator(T temperature,
                              T molecular_mass,
                              const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0)),
                              unsigned int seed               = atlas::seed::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    /**
     * @brief Generate a Maxwell-Boltzmann-distributed velocity sample on the host.
     *
     * @details
     * Produces a host-side sample using the stored generator parameters.
     *
     * @return Generated particle velocity.
     */
    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    /**
     * @brief Return a cached backend-portable generate operator.
     *
     * @details
     * Provides access to the internally cached operator corresponding to this
     * generator's current parameters.
     *
     * @return Const reference to the cached generate operator.
     */
    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

    /**
     * @brief Create a backend-portable generate operator by value.
     *
     * @details
     * Returns a by-value operator representing this generator's current
     * Maxwell-Boltzmann configuration.
     *
     * @return Generate operator corresponding to this generator.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    /**
     * @brief Return the primary scalar generator parameter.
     *
     * @details
     * For this generator, the primary parameter is the thermal temperature.
     *
     * @return Stored temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    /**
     * @brief Return the secondary scalar generator parameter.
     *
     * @details
     * For this generator, the secondary parameter is the molecular mass.
     *
     * @return Stored molecular mass.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    /**
     * @brief Return the runtime generation-law type.
     *
     * @return @ref GenerateType::maxwell_boltzmann.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    /**
     * @brief Stored thermal temperature.
     *
     * @details
     * Controls the thermal width of the Maxwell-Boltzmann distribution.
     */
    T _temperature;

    /**
     * @brief Stored molecular mass.
     *
     * @details
     * Controls the thermal spread of the generated velocity distribution.
     */
    T _molecular_mass;

    /**
     * @brief Stored mean drift velocity.
     *
     * @details
     * Shifts the generated velocity distribution by a bulk-motion offset.
     */
    Vector3<T> _bulk_velocity;

    /**
     * @brief Stored deterministic random seed.
     *
     * @details
     * Used to initialize the underlying random sampling logic.
     */
    unsigned int _seed;

    /**
     * @brief Cached backend-portable generate operator.
     *
     * @details
     * Stores a reusable operator representation corresponding to the current
     * generator parameters.
     */
    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

/**
 * @brief Fluent builder for @ref MaxwellBoltzmannGenerator.
 *
 * @details
 * This builder provides a controlled construction path for
 * @ref MaxwellBoltzmannGenerator by staging:
 * - temperature,
 * - molecular mass,
 * - bulk velocity,
 * - random seed.
 *
 * ## Typical usage
 * @code
 * auto generator = atlas::MaxwellBoltzmannGenerator<float>::builder()
 *     .with_temperature(300.0f)
 *     .with_molecular_mass(4.65e-26f)
 *     .with_bulk_velocity({0.0f, 0.0f, 10.0f})
 *     .with_seed(42u)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - temperature has been provided,
 * - molecular mass has been provided,
 * - temperature is physically admissible,
 * - molecular mass is positive and finite.
 *
 * The exact validation rules are implementation-defined in
 * `maxwell_boltzmann_generator.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class MaxwellBoltzmannGenerator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no staged temperature or molecular mass, zero bulk
     * velocity, and a default seed of `0`.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref MaxwellBoltzmannGenerator by value after validation.
     *
     * @return Fully constructed generator value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellBoltzmannGenerator<T>
    build() const;

    /**
     * @brief Build a configured @ref MaxwellBoltzmannGenerator in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>` owning the constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
    make_host_shared() const;

    /**
     * @brief Set the thermal temperature.
     *
     * @param temperature Staged temperature value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    /**
     * @brief Set the molecular mass.
     *
     * @param molecular_mass Staged molecular mass value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    /**
     * @brief Set the bulk velocity.
     *
     * @param bulk_velocity Staged mean drift velocity.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept;

    /**
     * @brief Set the deterministic random seed.
     *
     * @param seed Staged seed value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged physical parameters.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending thermal temperature.
     */
    std::optional<T> _temperature;

    /**
     * @brief Pending molecular mass.
     */
    std::optional<T> _molecular_mass;

    /**
     * @brief Pending bulk velocity.
     */
    Vector3<T> _bulk_velocity { T(0), T(0), T(0) };

    /**
     * @brief Pending deterministic random seed.
     */
    unsigned int _seed = atlas::seed::DEFAULT_UNSIGNED_INT_SEED;
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::fluid::MaxwellBoltzmannGenerateOperator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MaxwellBoltzmannGenerateOperator = atlas::fluid::MaxwellBoltzmannGenerateOperator<T>;

/**
 * @brief Convenience alias for @ref atlas::fluid::MaxwellBoltzmannGenerator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MaxwellBoltzmannGenerator = atlas::fluid::MaxwellBoltzmannGenerator<T>;

} // namespace atlas

#include <atlas/generator/maxwell_boltzmann_generator.hpp>
