#pragma once

#include <atlas/core/host_variant.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator_type.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <concepts>
#include <cstddef>
#include <utility>

namespace atlas {

template <typename G>
concept ConceptGenerator = requires(G generator,
                                    const G const_generator,
                                    FluidVelocityState* velocities,
                                    FluidSpeciesState* species,
                                    const Float3 bulk_velocity,
                                    std::size_t offset,
                                    std::size_t count) {
    { const_generator.generate(velocities, species, offset, count) } -> std::same_as<int>;
    { generator.set_bulk_velocity(bulk_velocity) } -> std::same_as<void>;
};

static_assert(ConceptGenerator<UniformGenerator>);
static_assert(ConceptGenerator<JitteringGenerator>);
static_assert(ConceptGenerator<MaxwellSigmaGenerator>);
static_assert(ConceptGenerator<MaxwellBoltzmannGenerator>);

class Generator final {
public:
    GeneratorType type = GeneratorType::uniform;

    union {

        UniformGenerator uniform;

        JitteringGenerator jittering;

        MaxwellSigmaGenerator maxwell_sigma;

        MaxwellBoltzmannGenerator maxwell_boltzmann;
    };

    ATLAS_HOST
    Generator() noexcept;

    ATLAS_HOST explicit Generator(UniformGenerator op) noexcept;

    ATLAS_HOST explicit Generator(JitteringGenerator op) noexcept;

    ATLAS_HOST explicit Generator(MaxwellSigmaGenerator op) noexcept;

    ATLAS_HOST explicit Generator(MaxwellBoltzmannGenerator op) noexcept;

    Generator(const Generator&) = delete;

    Generator&
    operator=(const Generator&)
        = delete;

    ATLAS_HOST
    Generator(Generator&& other) noexcept;

    ATLAS_HOST Generator&
    operator=(Generator&& other) noexcept;

    ATLAS_HOST
    ~Generator() noexcept;

    ATLAS_NODISCARD ATLAS_HOST int
    generate(FluidVelocityState* velocities,
             FluidSpeciesState* species,
             std::size_t offset,
             std::size_t count) const;

    ATLAS_HOST void
    set_bulk_velocity(const Float3& bulk_velocity) noexcept;
};

using GeneratorVariant = HostVariant<
    Generator,
    GeneratorType,
    GeneratorType::uniform,
    HostVariantCase<GeneratorType::uniform, &Generator::uniform>,
    HostVariantCase<GeneratorType::jittering, &Generator::jittering>,
    HostVariantCase<GeneratorType::maxwell_sigma, &Generator::maxwell_sigma>,
    HostVariantCase<GeneratorType::maxwell_boltzmann, &Generator::maxwell_boltzmann>>;

class GeneratorGenerate {
public:
    FluidVelocityState* velocities;
    FluidSpeciesState* species;
    std::size_t offset;
    std::size_t count;
    template <typename G>
    ATLAS_HOST int
    operator()(const G& generator) const {
        return generator.generate(velocities, species, offset, count);
    }
};

class GeneratorSetBulkVelocity {
public:
    const Float3& bulk_velocity;
    template <typename G>
    ATLAS_HOST void
    operator()(G& generator) const { generator.set_bulk_velocity(bulk_velocity); }
};

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator() noexcept {
    GeneratorVariant::construct(*this, GeneratorType::uniform);
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(UniformGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(JitteringGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(MaxwellSigmaGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(MaxwellBoltzmannGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(Generator&& other) noexcept {
    GeneratorVariant::move_construct(*this, std::move(other));
}

ATLAS_HOST ATLAS_FORCE_INLINE Generator&
Generator::operator=(Generator&& other) noexcept {
    GeneratorVariant::move_assign(*this, std::move(other));
    return *this;
}

ATLAS_HOST ATLAS_FORCE_INLINE
    Generator::~Generator() noexcept {
    GeneratorVariant::destroy(*this);
}

ATLAS_HOST ATLAS_FORCE_INLINE int
Generator::generate(FluidVelocityState* velocities,
                    FluidSpeciesState* species,
                    const std::size_t offset,
                    const std::size_t count) const {
    return GeneratorVariant::visit(
        *this,
        GeneratorGenerate { velocities, species, offset, count },
        0);
}

ATLAS_HOST ATLAS_FORCE_INLINE void
Generator::set_bulk_velocity(const Float3& bulk_velocity) noexcept {
    GeneratorVariant::apply(*this, GeneratorSetBulkVelocity { bulk_velocity });
}

using GeneratorHostPtr = atlas::host_shared_ptr<Generator>;

using GeneratorDevicePtr = atlas::device_shared_ptr<Generator>;

}