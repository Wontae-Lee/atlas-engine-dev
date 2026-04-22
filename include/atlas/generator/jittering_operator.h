#pragma once

/**
 * @file jittering_operator.h
 * @brief Declares jitter-based particle-value generation operators and host-side generator wrappers.
 */

#include <atlas/generator/generator.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

namespace atlas::fluid {

/**
 * @brief Device/host generation operator that produces a jittered 3D value.
 *
 * The operator generates each vector component independently by sampling a
 * uniform random offset in the range `[-jitter_radius, +jitter_radius]` and
 * adding it to `base_value`.
 *
 * @tparam T Scalar type used by the operator.
 */
template <typename T>
struct JitteringGenerateOperator final {
    unsigned int seed = static_cast<unsigned int>(atlas::seed::DEFAULT_UNSIGNED_INT_SEED);
    T base_value { T(0) };
    T jitter_radius { T(0) };
    mutable atlas::default_random_engine<T> engine;

    /**
     * @brief Construct a jittering generation operator.
     *
     * @param seed Seed used to initialize the internal random engine.
     * @param base_value Central value around which jitter is applied.
     * @param jitter_radius Radius of the symmetric uniform perturbation interval.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit JitteringGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::seed::DEFAULT_UNSIGNED_INT_SEED),
        T base_value      = T(0),
        T jitter_radius   = T(0)) noexcept;

    /**
     * @brief Generate a jittered 3D value.
     *
     * The input parameters are accepted to match the generic generate-operator
     * interface but are not used by this operator.
     *
     * @param param0 Unused.
     * @param param1 Unused.
     * @return Jittered 3D vector.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1) const;
};

/**
 * @brief Host-side generator wrapper for `JitteringGenerateOperator`.
 *
 * This class owns generator configuration, exposes the common `Generator<T>`
 * interface, and can construct both host-side and plain operator forms.
 *
 * @tparam T Scalar type used by the generator.
 */
template <typename T>
class JitteringOperator final : public Generator<T> {
public:
    class Builder;

public:
    /**
     * @brief Create a builder for `JitteringOperator`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Construct a jittering generator.
     *
     * @param base_value Central value around which jitter is applied.
     * @param jitter_radius Radius of the symmetric uniform perturbation interval.
     * @param seed Seed used to initialize the random engine.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    JitteringOperator(
        T base_value,
        T jitter_radius,
        unsigned int seed = atlas::seed::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    /**
     * @brief Generate a jittered vector value on the host.
     *
     * @return Generated vector.
     */
    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    /**
     * @brief Return the stored polymorphic generate operator.
     *
     * @return Reference to the internal generate operator.
     */
    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

    /**
     * @brief Return a value copy of the stored generate operator.
     *
     * @return Generate operator value.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    /**
     * @brief Return the first generator parameter.
     *
     * For this generator, the first parameter corresponds to `base_value`.
     *
     * @return Base value.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    /**
     * @brief Return the second generator parameter.
     *
     * For this generator, the second parameter corresponds to `jitter_radius`.
     *
     * @return Jitter radius.
     */
    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    /**
     * @brief Return the generator type tag.
     *
     * @return `GenerateType::jittering`.
     */
    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    T _base_value;
    T _jitter_radius;
    unsigned int _seed;
    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

/**
 * @brief Builder for `JitteringOperator`.
 *
 * The builder requires both `base_value` and `jitter_radius` to be provided
 * before construction.
 *
 * @tparam T Scalar type used by the generator.
 */
template <typename T>
class JitteringOperator<T>::Builder final {
public:
    Builder() = default;

    /**
     * @brief Set the central value used by the generator.
     *
     * @param base_value Central value around which jitter is applied.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_base_value(T base_value) noexcept;

    /**
     * @brief Set the jitter radius used by the generator.
     *
     * @param jitter_radius Radius of the symmetric perturbation interval.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_jitter_radius(T jitter_radius) noexcept;

    /**
     * @brief Set the random seed used by the generator.
     *
     * @param seed Seed used to initialize the random engine.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

    /**
     * @brief Build a validated `JitteringOperator`.
     *
     * @return Constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE JitteringOperator<T>
    build() const;

    /**
     * @brief Build a validated `JitteringOperator` and wrap it in a host-shared pointer.
     *
     * @return Host-shared pointer to the constructed generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<JitteringOperator<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the builder configuration.
     *
     * @throws std::runtime_error Thrown when required parameters are missing or invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _base_value;
    std::optional<T> _jitter_radius;
    unsigned int _seed = atlas::seed::DEFAULT_UNSIGNED_INT_SEED;
};

} // namespace atlas::fluid

namespace atlas {

template <typename T>
using JitteringOperator = atlas::fluid::JitteringOperator<T>;

} // namespace atlas

#include <atlas/generator/jittering_operator.hpp>