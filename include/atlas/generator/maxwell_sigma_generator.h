#pragma once

#include <atlas/generator/generate.h>
#include <atlas/generator/generator.h>
#include <atlas/random/seed.h>

#include <optional>

/**
 * @file maxwell_sigma_generator.h
 * @brief Host-side `Generator` wrapping `MaxwellSigmaGenerate`
 *        (`param0` = `sigma`); see `generate_payload.h` for the
 *        isotropic-Gaussian sampling model.
 */

namespace atlas {

/** @brief `Generator` configuration for `GenerateType::maxwell_sigma`. */
class MaxwellSigmaGenerator final : public Generator {
public:
    class Builder;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_HOST explicit MaxwellSigmaGenerator(
        float sigma,
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
    float _sigma;
    Generate _operator;
};

/** @brief Fluent builder for `MaxwellSigmaGenerator`; requires `_sigma`
 *  set. */
class MaxwellSigmaGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_sigma(float sigma) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_NODISCARD MaxwellSigmaGenerator
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<MaxwellSigmaGenerator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<float> _sigma;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

}
