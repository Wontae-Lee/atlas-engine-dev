#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/generator/generate_payload.h>
#include <atlas/random/seed.h>

#include <type_traits>

namespace atlas {

enum class GenerateType : int {

    uniform,

    jittering,

    maxwell_sigma,

    maxwell_boltzmann
};

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

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    generate(float param0,
             float param1 = 1.0f) const;

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
