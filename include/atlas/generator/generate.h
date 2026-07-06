#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/generator/generate_payload.h>
#include <atlas/random/seed.h>

#include <type_traits>

/**
 * @file generate.h
 * @brief Runtime-selectable dispatcher over the four velocity generators
 *        (`UniformGenerate`, `JitteringGenerate`,
 *        `MaxwellSigmaGenerate`, `MaxwellBoltzmannGenerate`),
 *        so a `Source`/`Generator` can be configured with any one of
 *        them at build time without templating every call site.
 *
 * @details
 * Follows the same tagged-union `DeviceVariant` pattern as
 * `DsmcKernel`/`SphKernel`/`SurfaceInteractionKernel` (see those files):
 * a `GenerateType` tag selects which payload generator is active, and
 * `detail::GenerateVariant::visit` dispatches `generate()` calls
 * to it. `generate(param0, param1)` maps generically onto each payload's
 * own parameter meaning — `(min_value, max_value)` for `uniform`,
 * ignored for `jittering`, `(sigma, unused)` for `maxwell_sigma`,
 * `(temperature, molecular_mass)` for `maxwell_boltzmann` — so
 * `detail::SourceEmitter` can call `generate(seed, temperature,
 * molecular_mass)` uniformly regardless of which generator a species
 * actually uses.
 */

namespace atlas {

/**
 * @brief Selects which velocity generator a `Generate` applies;
 *        see `generate_payload.h` for each generator's sampling model.
 */
enum class GenerateType : int {
    /** `UniformGenerate`. */
    uniform,
    /** `JitteringGenerate`. */
    jittering,
    /** `MaxwellSigmaGenerate`. */
    maxwell_sigma,
    /** `MaxwellBoltzmannGenerate`. */
    maxwell_boltzmann
};

/**
 * @brief Tagged-union wrapper letting `Source`/`Fluid::generators()`
 *        draw velocities through whichever `GenerateType` a species was
 *        configured with. See this file's top-of-file documentation for
 *        the `DeviceVariant` pattern and the generic `(param0, param1)`
 *        parameter mapping.
 */
struct Generate final {

    GenerateType type = GenerateType::uniform;

    union {

        UniformGenerate uniform;

        JitteringGenerate jittering;

        MaxwellSigmaGenerate maxwell_sigma;

        MaxwellBoltzmannGenerate maxwell_boltzmann;
    };

    ATLAS_ALL_DEVICE
    Generate() noexcept;

    ATLAS_ALL_DEVICE
    Generate(GenerateType type,
             unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_ALL_DEVICE
    Generate(const Generate& other) noexcept = default;

    ATLAS_ALL_DEVICE Generate&
    operator=(const Generate& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Generate() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Generate>, int> = 0>
    ATLAS_ALL_DEVICE explicit Generate(const Payload& op);

    /** @brief Dispatches to the active payload's `generate()` using its
     *  own internal (mutable, stateful) RNG engine. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(float param0,
             float param1 = 1.0f) const;

    /** @brief Dispatches to the active payload's `generate()` using a
     *  fresh engine seeded from `seed` — the stateless form used for
     *  independent per-thread device draws (see this file's
     *  top-of-file documentation). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(unsigned int seed,
             float param0,
             float param1 = 1.0f) const;
};

namespace detail {

    using GenerateVariant = DeviceVariant<
        Generate,
        GenerateType,
        GenerateType::uniform,
        DeviceVariantCase<GenerateType::uniform, &Generate::uniform>,
        DeviceVariantCase<GenerateType::jittering, &Generate::jittering>,
        DeviceVariantCase<GenerateType::maxwell_sigma, &Generate::maxwell_sigma>,
        DeviceVariantCase<GenerateType::maxwell_boltzmann, &Generate::maxwell_boltzmann>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
    struct GenerateSample {
        float param0;
        float param1;
        template <typename P>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
        operator()(const P& op) const { return op.generate(param0, param1); }
    };
    struct GenerateSampleSeeded {
        unsigned int seed;
        float param0;
        float param1;
        template <typename P>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
        operator()(const P& op) const { return op.generate(seed, param0, param1); }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Generate::Generate() noexcept {
    detail::GenerateVariant::construct(*this, GenerateType::uniform);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Generate::Generate(const GenerateType type,
                   const unsigned int seed) noexcept {
    detail::GenerateVariant::construct(*this, type, seed);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Generate>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Generate::Generate(const Payload& op) {
    detail::GenerateVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Generate::generate(const float param0,
                   const float param1) const {
    return detail::GenerateVariant::visit(
        *this,
        detail::GenerateSample { param0, param1 },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
Generate::generate(const unsigned int seed,
                   const float param0,
                   const float param1) const {
    return detail::GenerateVariant::visit(
        *this,
        detail::GenerateSampleSeeded { seed, param0, param1 },
        Float3(0.0f, 0.0f, 0.0f));
}

}
