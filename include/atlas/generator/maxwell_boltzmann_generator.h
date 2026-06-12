#pragma once

/**
 * @file maxwell_boltzmann_generator.h
 * @brief Declares Maxwell-Boltzmann particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::MaxwellBoltzmannGenerateOperator, a backend-portable
 *   operator that samples particle velocities according to a Maxwell-Boltzmann
 *   distribution, and
 * - @ref atlas::MaxwellBoltzmannGenerator, a host-side generator class
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
#include <atlas/generator/generate_operator.h>
#include <optional>

namespace atlas {

template <typename T>
class MaxwellBoltzmannGenerator final : public Generator<T> {
public:
    class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellBoltzmannGenerator(
        T temperature,
        T molecular_mass,
        const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0)),
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
    T _temperature;
    T _molecular_mass;
    Vector3<T> _bulk_velocity;
    unsigned int _seed;
    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

template <typename T>
class MaxwellBoltzmannGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellBoltzmannGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _temperature;
    std::optional<T> _molecular_mass;
    Vector3<T> _bulk_velocity { T(0), T(0), T(0) };
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

} // namespace atlas

#include <atlas/generator/maxwell_boltzmann_generator.hpp>
