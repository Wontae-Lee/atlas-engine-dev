#pragma once

#include <atlas/generator/generate.h>
#include <atlas/generator/generator.h>
#include <atlas/random/seed.h>

#include <optional>

namespace atlas {

class JitteringGenerator final : public Generator {
public:
    class Builder;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST
    JitteringGenerator(float base_value,
                       float jitter_radius,
                       unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Float3
    generate() const override;

    ATLAS_NODISCARD ATLAS_HOST const Generate&
    generate_operator() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST Generate
    make_generate_operator() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST float
    param0() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST float
    param1() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST GenerateType
    type() const noexcept override;

private:
    float _base_value;
    float _jitter_radius;
    Generate _operator;
};

class JitteringGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_base_value(float base_value) noexcept;

    ATLAS_HOST Builder&
    with_jitter_radius(float jitter_radius) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST JitteringGenerator
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<JitteringGenerator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<float> _base_value;
    std::optional<float> _jitter_radius;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

}
