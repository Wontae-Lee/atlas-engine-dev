#pragma once

/**
 * @file maxwell_sigma_generator.h
 * @brief Declares Maxwell-sigma particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::MaxwellSigmaGenerateOperator, a lightweight backend-portable
 *   operator for sampling velocity vectors from a Maxwell-style distribution with
 *   a shared standard deviation parameter, and
 * - @ref atlas::MaxwellSigmaGenerator, a host-side polymorphic generator
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
#include <atlas/generator/generate_operator.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

namespace atlas {

template <typename T>
class MaxwellSigmaGenerator final : public Generator<T> {
public:
    class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit
    MaxwellSigmaGenerator(
        T sigma,
        unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    T _sigma;
    unsigned int _seed;
    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

template <typename T>
class MaxwellSigmaGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sigma(T sigma) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellSigmaGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _sigma;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

} // namespace atlas

#include <atlas/generator/maxwell_sigma_generator.hpp>
