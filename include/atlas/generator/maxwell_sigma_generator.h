#pragma once

/**
 * @file maxwell_sigma_generator.h
 * @brief Declares a host-side Gaussian velocity generator parameterized by sigma.
 */

#include <atlas/generator/generator.h>

#include <optional>

namespace atlas::system {

/**
 * @brief Host-side generator for a Gaussian velocity distribution with shared sigma.
 */
template <typename T>
class MaxwellSigmaGenerator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellSigmaGenerator(T sigma, unsigned int seed = 0u) noexcept;

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
    T _sigma; ///< Shared standard deviation.
    unsigned int _seed; ///< Deterministic seed.
    GenerateOperator<T> _operator; ///< Cached backend-portable operator.
};

template <typename T>
class MaxwellSigmaGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellSigmaGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sigma(T sigma) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _sigma;
    unsigned int _seed = 0u;
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::MaxwellSigmaGenerator.
 */
template <typename T>
using MaxwellSigmaGenerator = atlas::system::MaxwellSigmaGenerator<T>;

}

#include <atlas/generator/maxwell_sigma_generator.hpp>
