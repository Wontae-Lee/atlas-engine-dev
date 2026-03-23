#pragma once

#include <atlas/generator/generator.h>

#include <optional>

namespace atlas::system {

template <typename T>
class MaxwellBoltzmannGenerator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellBoltzmannGenerator(T temperature,
                              T molecular_mass,
                              const Vector3<T>& bulk_velocity = Vector3<T>(T(0), T(0), T(0)),
                              unsigned int seed               = 0u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

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
    GenerateOperator<T> _operator;
};

template <typename T>
class MaxwellBoltzmannGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellBoltzmannGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _temperature;
    std::optional<T> _molecular_mass;
    Vector3<T> _bulk_velocity { T(0), T(0), T(0) };
    unsigned int _seed = 0u;
};

}

namespace atlas {

template <typename T>
using MaxwellBoltzmannGenerator = atlas::system::MaxwellBoltzmannGenerator<T>;

}

#include <atlas/generator/maxwell_boltzmann_generator.hpp>
