#pragma once

#include <atlas/math/math.h>

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/generator.h>
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
        unsigned int seed               = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

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
    GenerateOperator<T> _operator;
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

}

#include <atlas/generator/maxwell_boltzmann_generator.hpp>