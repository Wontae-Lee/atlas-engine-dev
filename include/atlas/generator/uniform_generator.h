#pragma once

/**
 * @file uniform_generator.h
 * @brief Declares uniform particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::UniformGenerateOperator, a backend-portable operator for
 *   sampling velocity vectors with independently uniform components, and
 * - @ref atlas::system::UniformGenerator, a host-side polymorphic generator that
 *   owns the corresponding runtime parameters and exports a portable
 *   @ref GenerateOperator.
 *
 * ## Distribution model
 * The uniform generator samples each velocity component independently from a
 * uniform distribution:
 * \f[
 * v_i \sim \mathcal{U}(\text{min}, \text{max})
 * \f]
 * for each axis \f$i \in \{x,y,z\}\f$.
 *
 * This is useful for:
 * - randomized initialization,
 * - bounded velocity injection,
 * - testing and debugging emission pipelines.
 *
 * ## Host/device split
 * Atlas separates this generation law into:
 * - a host-side @ref UniformGenerator used for configuration and direct sampling,
 * - a backend-friendly @ref UniformGenerateOperator embedded in
 *   @ref GenerateOperator for device-side emission.
 *
 * ## Construction
 * The host-side generator may be:
 * - constructed directly from minimum and maximum bounds, or
 * - built through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for parameters and generated velocities.
 */

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

#include <atlas/generator/generator.h>

namespace atlas::system {

/**
 * @brief Backend-portable operator for uniform velocity sampling.
 *
 * @details
 * @ref UniformGenerateOperator is a lightweight device-friendly representation
 * of a generator that samples velocity components uniformly within a given
 * range.
 *
 * It stores:
 * - a deterministic seed,
 * - a backend-portable random engine used for sampling.
 *
 * The operator is typically embedded inside a @ref GenerateOperator so that
 * emission code can evaluate it inside backend kernels.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct UniformGenerateOperator final {
    /**
     * @brief Deterministic seed used to initialize the random engine.
     */
    unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed);

    /**
     * @brief Internal pseudo-random engine used for sampling.
     *
     * @details
     * The exact random-number generation behavior is defined by
     * @ref atlas::default_random_engine.
     */
    mutable atlas::default_random_engine<T> engine;

    /**
     * @brief Construct a uniform generation operator.
     *
     * @param seed Deterministic seed for reproducible sampling.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit UniformGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed)) noexcept;

    /**
     * @brief Generate a velocity sample with uniformly distributed components.
     *
     * @details
     * Each component of the returned vector is independently sampled from the
     * interval \f$[\text{min\_value}, \text{max\_value}]\f$.
     *
     * @param min_value Lower bound for each velocity component.
     * @param max_value Upper bound for each velocity component.
     * @return Generated particle velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T min_value,
             T max_value) const;
};

/**
 * @brief Host-side generator for uniformly distributed velocity components.
 *
 * @details
 * @ref UniformGenerator is the polymorphic host-side representation of the
 * uniform velocity generation law.
 *
 * It stores:
 * - a minimum bound,
 * - a maximum bound,
 * - a deterministic random seed,
 * - a cached backend-portable @ref GenerateOperator.
 *
 * ## Responsibilities
 * A generator instance can:
 * - produce host-side velocity samples via @ref generate,
 * - expose a cached backend operator via @ref generate_operator,
 * - create a backend operator by value via @ref make_generate_operator,
 * - report scalar parameters via @ref param0 and @ref param1,
 * - report its runtime type via @ref type.
 *
 * ## Parameter mapping
 * For this generator:
 * - @ref param0 corresponds to the minimum bound,
 * - @ref param1 corresponds to the maximum bound.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class UniformGenerator final : public Generator<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing
     *        @ref UniformGenerator.
     *
     * @details
     * The builder stages:
     * - minimum and maximum bounds,
     * - deterministic seed,
     * validates them, and constructs either:
     * - a value instance, or
     * - a host-shared pointer.
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
     * @brief Construct a uniform generator from explicit bounds and seed.
     *
     * @param min_value Minimum component value.
     * @param max_value Maximum component value.
     * @param seed Deterministic random seed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    UniformGenerator(
        T min_value,
        T max_value,
        unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed)) noexcept;

    /**
     * @brief Generate a uniformly distributed velocity sample on the host.
     *
     * @return Generated particle velocity.
     */
    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    /**
     * @brief Return a cached backend-portable generate operator.
     *
     * @return Const reference to the cached generate operator.
     */
    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

    /**
     * @brief Create a backend-portable generate operator by value.
     *
     * @return Generate operator corresponding to this generator.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    /**
     * @brief Return the minimum bound parameter.
     *
     * @return Minimum component value.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    /**
     * @brief Return the maximum bound parameter.
     *
     * @return Maximum component value.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    /**
     * @brief Return the runtime generation-law type.
     *
     * @return @ref GenerateType::uniform.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    /**
     * @brief Minimum component value.
     */
    T _min_value;

    /**
     * @brief Maximum component value.
     */
    T _max_value;

    /**
     * @brief Deterministic random seed.
     */
    unsigned int _seed;

    /**
     * @brief Cached backend-portable generate operator.
     */
    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

/**
 * @brief Fluent builder for @ref UniformGenerator.
 *
 * @details
 * Provides a controlled construction path for @ref UniformGenerator by staging:
 * - minimum bound,
 * - maximum bound,
 * - deterministic seed.
 *
 * ## Typical usage
 * @code
 * auto generator = atlas::UniformGenerator<float>::builder()
 *     .with_min_value(-1.0f)
 *     .with_max_value(1.0f)
 *     .with_seed(42u)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked before construction. Typical checks include:
 * - both bounds are provided,
 * - min_value <= max_value,
 * - bounds are finite.
 *
 * The exact validation behavior is implementation-defined.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class UniformGenerator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref UniformGenerator by value.
     *
     * @return Fully constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE UniformGenerator<T>
    build() const;

    /**
     * @brief Build a configured @ref UniformGenerator in a host_shared_ptr.
     *
     * @return Shared pointer to constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<UniformGenerator<T>>
    make_host_shared() const;

    /**
     * @brief Set the minimum bound.
     *
     * @param min_value Lower bound.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_min_value(T min_value) noexcept;

    /**
     * @brief Set the maximum bound.
     *
     * @param max_value Upper bound.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_max_value(T max_value) noexcept;

    /**
     * @brief Set the deterministic seed.
     *
     * @param seed Seed value.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending minimum bound.
     */
    std::optional<T> _min_value;

    /**
     * @brief Pending maximum bound.
     */
    std::optional<T> _max_value;

    /**
     * @brief Pending deterministic seed.
     */
    unsigned int _seed = atlas::seed::default_unsigned_int_seed;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::UniformGenerateOperator.
 */
template <typename T>
using UniformGenerateOperator = atlas::system::UniformGenerateOperator<T>;

/**
 * @brief Convenience alias for @ref atlas::system::UniformGenerator.
 */
template <typename T>
using UniformGenerator = atlas::system::UniformGenerator<T>;

} // namespace atlas

#include <atlas/generator/uniform_generator.hpp>
