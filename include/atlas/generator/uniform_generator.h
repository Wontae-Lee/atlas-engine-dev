#pragma once

#include <atlas/generator/generator.h>

#include <optional>

namespace atlas::system {

template <typename T>
class UniformGenerator final : public Generator<T> {
public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    UniformGenerator(T min_value, T max_value, unsigned int seed = 0u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    T _min_value;
    T _max_value;
    unsigned int _seed;
    UniformGenerateOperator<T> _operator;
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
    unsigned int _seed = 0u;
};

}

namespace atlas {

template <typename T>
using UniformGenerator = atlas::system::UniformGenerator<T>;

}

#include <atlas/generator/uniform_generator.hpp>