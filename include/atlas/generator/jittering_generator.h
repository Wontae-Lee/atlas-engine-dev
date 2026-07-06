#pragma once

#include <atlas/generator/generate.h>
#include <atlas/generator/generator.h>
#include <atlas/random/seed.h>

#include <optional>

/**
 * @file jittering_generator.h
 * @brief Host-side `Generator` wrapping `JitteringGenerate`
 *        (`param0` = `base_value`, `param1` = `jitter_radius`); see
 *        `generate_payload.h` for why this is a non-physical,
 *        near-deterministic generator rather than a Maxwellian one.
 */

namespace atlas {

/** @brief `Generator` configuration for `GenerateType::jittering`. */
class JitteringGenerator final : public Generator {
public:
    class Builder;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_HOST
    JitteringGenerator(float base_value,
                      float jitter_radius,
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
    float _base_value;
    float _jitter_radius;
    Generate _operator;
};

/** @brief Fluent builder for `JitteringGenerator`; requires both
 *  `_base_value`/`_jitter_radius` set. */
class JitteringGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_base_value(float base_value) noexcept;

    ATLAS_HOST Builder&
    with_jitter_radius(float jitter_radius) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_NODISCARD JitteringGenerator
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<JitteringGenerator>
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
