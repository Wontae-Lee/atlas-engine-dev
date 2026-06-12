#pragma once

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/generator.h>
#include <atlas/math/math.h>

#include <optional>

namespace atlas {

template <typename T>
class MaxwellSigmaGenerator final : public Generator<T> {
public:
    class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerator(
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
    GenerateOperator<T> _operator;
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

}

#include <atlas/generator/maxwell_sigma_generator.hpp>