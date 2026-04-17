#pragma once

/**
 * @file maxwell_sigma_generator.h
 * @brief Declares Maxwell-sigma particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::MaxwellSigmaGenerateOperator, a lightweight backend-portable
 *   operator for sampling velocity vectors from a Maxwell-style distribution with
 *   a shared standard deviation parameter, and
 * - @ref atlas::system::MaxwellSigmaGenerator, a host-side polymorphic generator
 *   that owns the corresponding runtime parameters and can export a portable
 *   @ref GenerateOperator.
 *
 * ## Distribution model
 * The Maxwell-sigma generator uses a single scalar parameter, \f$\sigma\f$, to
 * control the spread of the sampled velocity distribution. In practice, this is
 * useful when the emission law is naturally expressed in terms of a shared
 * Gaussian scale rather than explicit thermodynamic parameters such as temperature
 * and molecular mass.
 *
 * ## Host/device split
 * Atlas separates this generation law into two complementary forms:
 * - a host-side @ref MaxwellSigmaGenerator used for runtime configuration,
 *   introspection, and direct host-side sampling,
 * - a backend-friendly @ref MaxwellSigmaGenerateOperator that can be embedded
 *   inside @ref GenerateOperator and executed in backend kernels.
 *
 * ## Construction
 * The host-side generator may be:
 * - constructed directly from `sigma` and an optional seed, or
 * - built through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for parameters and generated velocities.
 */

#include <atlas/generator/generator.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

namespace atlas::system {

/**
 * @brief Backend-portable operator for sampling Maxwell-style velocities with a shared sigma.
 *
 * @details
 * @ref MaxwellSigmaGenerateOperator is the compact device-friendly representation
 * of a velocity generator parameterized by a single shared standard deviation.
 *
 * It stores:
 * - a deterministic seed used to initialize its internal random engine,
 * - a backend-portable random engine used to produce pseudo-random samples.
 *
 * The operator is typically wrapped by @ref GenerateOperator so that source
 * emission code can store multiple generator laws in a uniform representation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct MaxwellSigmaGenerateOperator final {
    /**
     * @brief Deterministic seed used to initialize the random engine.
     *
     * @details
     * Controls reproducibility of generated samples.
     */
    unsigned int seed = atlas::seed::default_unsigned_int_seed;

    /**
     * @brief Internal pseudo-random engine used for velocity sampling.
     *
     * @details
     * The exact random-number generation behavior is defined by
     * @ref atlas::default_random_engine.
     */
    mutable atlas::default_random_engine<T> engine;

    /**
     * @brief Construct a Maxwell-sigma generation operator.
     *
     * @details
     * Initializes the operator with the supplied deterministic seed.
     *
     * @param seed Deterministic seed used for reproducible sampling.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerateOperator(
        unsigned int seed = atlas::seed::default_unsigned_int_seed) noexcept;

    /**
     * @brief Generate a velocity sample using the supplied shared sigma.
     *
     * @details
     * Produces a velocity vector whose distribution width is controlled by the
     * scalar parameter @p sigma.
     *
     * The exact sampling rule is implementation-defined in
     * `maxwell_sigma_generator.hpp`.
     *
     * @param sigma Shared standard deviation controlling distribution spread.
     * @return Generated particle velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T sigma) const;
};

/**
 * @brief Host-side generator for a Maxwell-style velocity distribution with shared sigma.
 *
 * @details
 * @ref MaxwellSigmaGenerator is the polymorphic host-side representation of the
 * Maxwell-sigma particle-velocity generation law.
 *
 * It stores:
 * - a shared standard deviation parameter,
 * - a deterministic random seed,
 * - a cached backend-portable @ref GenerateOperator corresponding to the current
 *   configuration.
 *
 * ## Responsibilities
 * A generator instance can:
 * - produce host-side velocity samples through @ref generate,
 * - expose its cached backend-portable operator through @ref generate_operator,
 * - create an operator by value through @ref make_generate_operator,
 * - report its scalar parameters through @ref param0 and @ref param1,
 * - report its runtime type through @ref type.
 *
 * ## Parameter mapping
 * For this generator:
 * - @ref param0 corresponds to `sigma`,
 * - @ref param1 is typically unused or implementation-defined.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class MaxwellSigmaGenerator final : public Generator<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing
     *        @ref MaxwellSigmaGenerator.
     *
     * @details
     * The builder stages `sigma` and the deterministic seed, validates them,
     * and then constructs either:
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
     * @brief Construct a Maxwell-sigma generator from explicit parameters.
     *
     * @param sigma Shared standard deviation controlling distribution width.
     * @param seed Deterministic random seed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellSigmaGenerator(
        T sigma,
        unsigned int seed = atlas::seed::default_unsigned_int_seed) noexcept;

    /**
     * @brief Generate a Maxwell-sigma-distributed velocity sample on the host.
     *
     * @details
     * Produces a host-side sample using the stored sigma and seed configuration.
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
     * generator's current configuration.
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
     * Maxwell-sigma configuration.
     *
     * @return Generate operator corresponding to this generator.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    /**
     * @brief Return the primary scalar generator parameter.
     *
     * @details
     * For this generator, the primary parameter is the shared standard deviation.
     *
     * @return Stored sigma.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    /**
     * @brief Return the secondary scalar generator parameter.
     *
     * @details
     * This generator is fundamentally parameterized by a single scalar value.
     * The secondary parameter is therefore typically unused or set according to
     * the implementation policy.
     *
     * @return Secondary scalar parameter.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    /**
     * @brief Return the runtime generation-law type.
     *
     * @return @ref GenerateType::maxwell_sigma.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    /**
     * @brief Stored shared standard deviation.
     *
     * @details
     * Controls the width of the generated velocity distribution.
     */
    T _sigma;

    /**
     * @brief Stored deterministic random seed.
     *
     * @details
     * Used to initialize the random sampling logic.
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
 * @brief Fluent builder for @ref MaxwellSigmaGenerator.
 *
 * @details
 * This builder provides a controlled construction path for
 * @ref MaxwellSigmaGenerator by staging:
 * - the shared standard deviation `sigma`,
 * - the deterministic random seed.
 *
 * ## Typical usage
 * @code
 * auto generator = atlas::MaxwellSigmaGenerator<float>::builder()
 *     .with_sigma(120.0f)
 *     .with_seed(42u)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - sigma has been provided,
 * - sigma is finite,
 * - sigma is non-negative or strictly positive according to implementation policy.
 *
 * The exact validation rules are implementation-defined in
 * `maxwell_sigma_generator.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class MaxwellSigmaGenerator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no staged sigma and a default seed of `0`.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref MaxwellSigmaGenerator by value after validation.
     *
     * @return Fully constructed generator value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellSigmaGenerator<T>
    build() const;

    /**
     * @brief Build a configured @ref MaxwellSigmaGenerator in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>` owning the constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
    make_host_shared() const;

    /**
     * @brief Set the shared standard deviation.
     *
     * @param sigma Staged sigma value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sigma(T sigma) noexcept;

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
     * Performs pre-construction checks on the staged sigma and seed values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending shared standard deviation.
     */
    std::optional<T> _sigma;

    /**
     * @brief Pending deterministic random seed.
     */
    unsigned int _seed = atlas::seed::default_unsigned_int_seed;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::MaxwellSigmaGenerateOperator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MaxwellSigmaGenerateOperator = atlas::system::MaxwellSigmaGenerateOperator<T>;

/**
 * @brief Convenience alias for @ref atlas::system::MaxwellSigmaGenerator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MaxwellSigmaGenerator = atlas::system::MaxwellSigmaGenerator<T>;

} // namespace atlas

#include <atlas/generator/maxwell_sigma_generator.hpp>
