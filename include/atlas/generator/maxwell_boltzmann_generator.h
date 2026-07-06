#pragma once

#include <atlas/generator/generate.h>
#include <atlas/generator/generator.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>

#include <optional>

/**
 * @file maxwell_boltzmann_generator.h
 * @brief Host-side `Generator` wrapping
 *        `MaxwellBoltzmannGenerate` (`param0` = `temperature`,
 *        `param1` = `molecular_mass`, plus a configured
 *        `bulk_velocity`); see `generate_payload.h` for the equilibrium
 *        kinetic-theory derivation. The physically standard choice for
 *        `Source` particle emission.
 */

namespace atlas {

/** @brief `Generator` configuration for `GenerateType::maxwell_boltzmann`. */
class MaxwellBoltzmannGenerator final : public Generator {
public:
    class Builder;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST
    MaxwellBoltzmannGenerator(
        float temperature,
        float molecular_mass,
        const Float3& bulk_velocity = Float3(0.0f, 0.0f, 0.0f),
        unsigned int seed           = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

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
    float _temperature;
    float _molecular_mass;
    Float3 _bulk_velocity;
    Generate _operator;
};

/** @brief Fluent builder for `MaxwellBoltzmannGenerator`; requires both
 *  `_temperature`/`_molecular_mass` set; `_bulk_velocity` defaults to
 *  zero. */
class MaxwellBoltzmannGenerator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_HOST Builder&
    with_molecular_mass(float molecular_mass) noexcept;

    ATLAS_HOST Builder&
    with_bulk_velocity(const Float3& bulk_velocity) noexcept;

    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellBoltzmannGenerator
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellBoltzmannGenerator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<float> _temperature;
    std::optional<float> _molecular_mass;
    Float3 _bulk_velocity = Float3(0.0f, 0.0f, 0.0f);
    unsigned int _seed    = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

}
