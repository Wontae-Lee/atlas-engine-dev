#pragma once

#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <optional>

#include <atlas/generator/generator.h>

namespace atlas::fluid {

template <typename T>
struct UniformGenerateOperator final {

    unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed);

    mutable atlas::default_random_engine<T> engine;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit UniformGenerateOperator(
        unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T min_value,
             T max_value) const;
};

template <typename T>
class UniformGenerator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    UniformGenerator(
        T min_value,
        T max_value,
        unsigned int seed = static_cast<unsigned int>(atlas::seed::default_unsigned_int_seed)) noexcept;

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
    T _min_value;

    T _max_value;

    unsigned int _seed;

    atlas::host_shared_ptr<GenerateOperator<T>> _operator;
};

template <typename T>
class UniformGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE UniformGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<UniformGenerator<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_min_value(T min_value) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_max_value(T max_value) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _min_value;

    std::optional<T> _max_value;

    unsigned int _seed = atlas::seed::default_unsigned_int_seed;
};

}

namespace atlas {

template <typename T>
using UniformGenerateOperator = atlas::fluid::UniformGenerateOperator<T>;

template <typename T>
using UniformGenerator = atlas::fluid::UniformGenerator<T>;

}

#include <atlas/generator/uniform_generator.hpp>