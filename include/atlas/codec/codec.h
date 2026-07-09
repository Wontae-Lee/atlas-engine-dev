#pragma once

#include <atlas/codec/codec_type.h>
#include <atlas/codec/knudsen_codec.h>
#include <atlas/core/host_variant.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <concepts>
#include <utility>

namespace atlas {

template <typename C>
concept ConceptCodec = requires(const C codec,
                                const UniverseTemperatureState* temperature,
                                const UniverseNumberParticleState* number_particle,
                                UniverseAllocatedSolverState* allocated_solver) {
    { codec.allocate(temperature, number_particle, allocated_solver) } -> std::same_as<void>;
};

static_assert(ConceptCodec<KnudsenCodec>);

// Host-side tagged union over the concrete codecs. Each leaf owns DeviceBuffer
// tables, so this uses HostVariant (host-only, move-based) rather than a
// DeviceVariant.
class Codec final {
public:
    CodecType type = CodecType::knudsen;

    union {

        KnudsenCodec knudsen;
    };

    ATLAS_HOST
    Codec() noexcept;

    ATLAS_HOST explicit
    Codec(KnudsenCodec op) noexcept;

    Codec(const Codec&) = delete;

    Codec&
    operator=(const Codec&)
        = delete;

    ATLAS_HOST
    Codec(Codec&& other) noexcept;

    ATLAS_HOST Codec&
    operator=(Codec&& other) noexcept;

    ATLAS_HOST
    ~Codec() noexcept;

    ATLAS_HOST void
    allocate(const UniverseTemperatureState* temperature,
             const UniverseNumberParticleState* number_particle,
             UniverseAllocatedSolverState* allocated_solver) const;
};

using CodecVariant = HostVariant<
    Codec,
    CodecType,
    CodecType::knudsen,
    HostVariantCase<CodecType::knudsen, &Codec::knudsen>>;

class CodecAllocate {
public:
    const UniverseTemperatureState* temperature;
    const UniverseNumberParticleState* number_particle;
    UniverseAllocatedSolverState* allocated_solver;
    template <typename C>
    ATLAS_HOST void
    operator()(const C& codec) const {
        codec.allocate(temperature, number_particle, allocated_solver);
    }
};

ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec() noexcept {
    CodecVariant::construct(*this, CodecType::knudsen);
}

ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec(KnudsenCodec op) noexcept {
    CodecVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec(Codec&& other) noexcept {
    CodecVariant::move_construct(*this, std::move(other));
}

ATLAS_HOST ATLAS_FORCE_INLINE Codec&
Codec::operator=(Codec&& other) noexcept {
    CodecVariant::move_assign(*this, std::move(other));
    return *this;
}

ATLAS_HOST ATLAS_FORCE_INLINE
Codec::~Codec() noexcept {
    CodecVariant::destroy(*this);
}

ATLAS_HOST ATLAS_FORCE_INLINE void
Codec::allocate(const UniverseTemperatureState* temperature,
                const UniverseNumberParticleState* number_particle,
                UniverseAllocatedSolverState* allocated_solver) const {
    CodecVariant::apply(
        *this,
        CodecAllocate { temperature, number_particle, allocated_solver });
}

using CodecHostPtr = atlas::host_shared_ptr<Codec>;

using CodecDevicePtr = atlas::device_shared_ptr<Codec>;

}
