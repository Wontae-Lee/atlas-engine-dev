#pragma once

#include <atlas/generator/generate.h>
#include <atlas/generator/generator.h>
#include <atlas/random/seed.h>

#include <optional>

/**
 * @file uniform_generator.h
 * @brief Host-side `Generator` wrapping `UniformGenerate`
 *        (`param0` = `min_value`, `param1` = `max_value`); see
 *        `generate_payload.h` for the sampling model.
 */

namespace atlas {

/** @brief `Generator` configuration for `GenerateType::uniform`. */
class UniformGenerator final : public Generator {
public:
    class Builder;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_HOST
    UniformGenerator(float min_value,
                     float max_value,
                     unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3
    generate() const override;

    ATLAS_HOST ATLAS_NODISCARD const Generate&
    generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD Generate
    make_generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD float
    param0() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD float
    param1() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    float _min_value;
    float _max_value;
    Generate _operator;
};

/** @brief Fluent builder for `UniformGenerator`; requires both
 *  `_min_value`/`_max_value` set. */
class UniformGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_min_value(float min_value) noexcept;

    ATLAS_HOST Builder&
    with_max_value(float max_value) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_NODISCARD UniformGenerator
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<UniformGenerator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<float> _min_value;
    std::optional<float> _max_value;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

}
