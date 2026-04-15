#pragma once

#include <atlas/generator/generator.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

namespace atlas::system {

template <typename T>
struct MaxwellSigmaGenerateOperator final {

    unsigned int seed = atlas::seed::default_unsigned_int_seed;

    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MaxwellSigmaGenerateOperator(
        unsigned int seed = atlas::seed::default_unsigned_int_seed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T sigma) const;
};

template <typename T>
class MaxwellSigmaGenerator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    MaxwellSigmaGenerator(
        T sigma,
        unsigned int seed = atlas::seed::default_unsigned_int_seed) noexcept;

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

    unsigned int _seed = atlas::seed::default_unsigned_int_seed;
};

}

namespace atlas {

template <typename T>
using MaxwellSigmaGenerateOperator = atlas::system::MaxwellSigmaGenerateOperator<T>;

template <typename T>
using MaxwellSigmaGenerator = atlas::system::MaxwellSigmaGenerator<T>;

}

#include <atlas/generator/maxwell_sigma_generator.hpp>