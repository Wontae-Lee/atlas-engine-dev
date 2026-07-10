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

/**
 * @brief Compile-time contract every codec leaf must satisfy to join the `Codec` union.
 *
 * A model must expose an `allocate` taking the three universe-state pointers by the
 * exact signature below and returning `void`. The `static_assert` under this concept
 * turns a missing or mistyped `allocate` on any leaf into a build error rather than a
 * silent dispatch failure.
 *
 * @tparam C Candidate leaf type (e.g. `KnudsenCodec`).
 */
template <typename C>
concept ConceptCodec = requires(const C codec,
                                const UniverseTemperatureState* temperature,
                                const UniverseNumberParticleState* number_particle,
                                UniverseAllocatedSolverState* allocated_solver) {
    { codec.allocate(temperature, number_particle, allocated_solver) } -> std::same_as<void>;
};

/// Guarantee the only current leaf conforms to the codec contract.
static_assert(ConceptCodec<KnudsenCodec>);

/**
 * @brief Tagged-union umbrella over the codec leaves; picks a per-cell solver.
 *
 * A codec reads the universe's per-cell states and writes, per cell, the index of
 * the solver `System::solve()` should run there. `Codec` holds exactly one leaf in
 * an anonymous union, discriminated by `type`, and forwards `allocate()` to it
 * through `HostVariant`'s tag dispatch.
 *
 * Like `Source` and `Generator`, the umbrella is a **move-only `HostVariant`**
 * (host-only, move-based) rather than a `DeviceVariant`, following the shared leaf
 * pattern for these owning umbrellas — copy is deleted and the lifecycle
 * (construct/move/destroy) is delegated to `CodecVariant`. Construct from a leaf,
 * e.g. `Codec c(KnudsenCodec::builder()...build());`.
 *
 * @note The leaf itself (`KnudsenCodec`) is trivially copyable and captured by value
 *       on the device inside its own `allocate`; the `HostVariant` choice governs the
 *       umbrella's lifecycle, not where the leaf runs.
 * @see ConceptCodec, CodecType, KnudsenCodec
 */
class Codec final {
public:
    /// Active-leaf discriminant; selects which union member and which dispatch case is live.
    CodecType type = CodecType::knudsen;

    /// Storage for exactly one leaf; only the member named by `type` is alive.
    union {

        /// The Knudsen-number codec leaf (active when `type == CodecType::knudsen`).
        KnudsenCodec knudsen;
    };

    /// Default-construct the default leaf (`KnudsenCodec`) via `CodecVariant`.
    ATLAS_HOST
    Codec() noexcept;

    /**
     * @brief Construct holding a moved-in `KnudsenCodec` leaf.
     * @param op Leaf to adopt; moved into the union.
     */
    ATLAS_HOST explicit Codec(KnudsenCodec op) noexcept;

    /// Deleted: the umbrella is move-only because its leaves are not copied.
    Codec(const Codec&) = delete;

    /// Deleted: see the copy constructor.
    Codec&
    operator=(const Codec&)
        = delete;

    /**
     * @brief Move-construct, adopting @p other's active leaf and leaving it valid.
     * @param other Source umbrella, consumed.
     */
    ATLAS_HOST
    Codec(Codec&& other) noexcept;

    /**
     * @brief Move-assign, destroying the current leaf and adopting @p other's.
     * @param other Source umbrella, consumed.
     * @return `*this`.
     */
    ATLAS_HOST Codec&
    operator=(Codec&& other) noexcept;

    /// Destroy the active leaf through `CodecVariant`.
    ATLAS_HOST
    ~Codec() noexcept;

    /**
     * @brief Dispatch `allocate` to the active leaf, filling per-cell solver indices.
     *
     * Forwards the three universe-state pointers unchanged to whichever leaf `type`
     * selects. Null-handling and no-op conditions are the leaf's responsibility (see
     * `KnudsenCodec::allocate`).
     *
     * @param temperature   Per-cell temperature state; may be unused by the leaf.
     * @param number_particle Per-cell particle counts (input).
     * @param allocated_solver Per-cell solver indices (output), overwritten in place.
     */
    ATLAS_HOST void
    allocate(const UniverseTemperatureState* temperature,
             const UniverseNumberParticleState* number_particle,
             UniverseAllocatedSolverState* allocated_solver) const;
};

/**
 * @brief `HostVariant` specialization wiring `Codec`'s tag to its union member.
 *
 * Binds `CodecType::knudsen` to `&Codec::knudsen` and names `knudsen` as the default
 * leaf. Every lifecycle and dispatch operation on `Codec` routes through this alias;
 * adding a leaf means adding a matching `HostVariantCase` here.
 */
using CodecVariant = HostVariant<
    Codec,
    CodecType,
    CodecType::knudsen,
    HostVariantCase<CodecType::knudsen, &Codec::knudsen>>;

/**
 * @brief Visitor that invokes `allocate` on whichever leaf `HostVariant::apply` selects.
 *
 * Carries the three universe-state pointers and applies them uniformly to any leaf
 * type `C`, letting `Codec::allocate` stay leaf-agnostic. Exists as a named struct
 * (rather than a lambda) so `HostVariant::apply` can call it as a generic visitor.
 */
class CodecAllocate {
public:
    /// Per-cell temperature state forwarded to the leaf (may be unused by it).
    const UniverseTemperatureState* temperature;

    /// Per-cell particle counts forwarded to the leaf (input).
    const UniverseNumberParticleState* number_particle;

    /// Per-cell solver indices forwarded to the leaf (output).
    UniverseAllocatedSolverState* allocated_solver;

    /**
     * @brief Call `allocate` on the visited leaf with the carried pointers.
     * @tparam C The concrete leaf type resolved by tag dispatch.
     * @param codec The active leaf.
     */
    template <typename C>
    ATLAS_HOST void
    operator()(const C& codec) const {
        codec.allocate(temperature, number_particle, allocated_solver);
    }
};

// Default-construct the tag's default leaf in place; the union starts alive and valid.
ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec() noexcept {
    CodecVariant::construct(*this, CodecType::knudsen);
}

// Set `type` from the payload's leaf and move-construct it into the union member.
ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec(KnudsenCodec op) noexcept {
    CodecVariant::construct_payload(*this, std::move(op));
}

// Move-construct the active leaf from `other`; `other` keeps a valid, destructible state.
ATLAS_HOST ATLAS_FORCE_INLINE
Codec::Codec(Codec&& other) noexcept {
    CodecVariant::move_construct(*this, std::move(other));
}

// Destroy the current leaf, then move-construct `other`'s in its place.
ATLAS_HOST ATLAS_FORCE_INLINE Codec&
Codec::operator=(Codec&& other) noexcept {
    CodecVariant::move_assign(*this, std::move(other));
    return *this;
}

// Run the active leaf's destructor through the variant's tag dispatch.
ATLAS_HOST ATLAS_FORCE_INLINE
    Codec::~Codec() noexcept {
    CodecVariant::destroy(*this);
}

// Wrap the pointers in a visitor and let `apply` route to the live leaf's `allocate`.
ATLAS_HOST ATLAS_FORCE_INLINE void
Codec::allocate(const UniverseTemperatureState* temperature,
                const UniverseNumberParticleState* number_particle,
                UniverseAllocatedSolverState* allocated_solver) const {
    CodecVariant::apply(
        *this,
        CodecAllocate { temperature, number_particle, allocated_solver });
}

/// Host-side shared owner of a `Codec` umbrella.
using CodecHostPtr = atlas::host_shared_ptr<Codec>;

/// Device-side shared owner of a `Codec` umbrella.
using CodecDevicePtr = atlas::device_shared_ptr<Codec>;

}