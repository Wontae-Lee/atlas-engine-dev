#pragma once

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/generator.h>

#include <optional>

namespace atlas {

template <typename T>
class JitteringOperator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    JitteringOperator(
        T base_value,
        T jitter_radius,
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
    T _base_value;
    T _jitter_radius;
    GenerateOperator<T> _operator;
};

template <typename T>
class JitteringOperator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_base_value(T base_value) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_jitter_radius(T jitter_radius) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE JitteringOperator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<JitteringOperator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _base_value;
    std::optional<T> _jitter_radius;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

}

#include <atlas/generator/jittering_operator.hpp>