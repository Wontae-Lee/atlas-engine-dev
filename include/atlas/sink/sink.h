#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sink/sink_type.h>
#include <atlas/sink/surface_sink.h>
#include <atlas/sink/tracing_sink.h>
#include <atlas/sink/volume_sink.h>

#include <concepts>
#include <type_traits>

namespace atlas {

/**
 * @brief Compile-time contract every @ref Sink leaf type must satisfy.
 *
 * A conforming leaf decides, per particle, whether it should be despawned at a
 * boundary and can advance its own boundary in time. The required surface is:
 *  - `bool despawn(const Float3& position, const Float3& velocity, float dt) const`
 *    — the removal predicate;
 *  - `void advance(float dt)` — step the leaf's owned unit forward.
 *
 * @tparam S The candidate leaf type.
 */
template <typename S>
concept ConceptSink = requires(S sink, const Float3 vec, float dt) {
    { sink.despawn(vec, vec, dt) } -> std::same_as<bool>;
    { sink.advance(dt) } -> std::same_as<void>;
};

/// Each concrete leaf is checked against the contract at compile time.
static_assert(ConceptSink<SurfaceSink>);
static_assert(ConceptSink<VolumeSink>);
static_assert(ConceptSink<TracingSink>);

/**
 * @brief Tagged-union umbrella wrapping one sink leaf and dispatching to it.
 *
 * A @ref Sink holds a @ref type tag and an anonymous union of the leaf types;
 * exactly the member named by @ref type is active. Because every leaf is trivially
 * copyable, @ref Sink itself is trivially copyable and uses `DeviceVariant`: it can
 * live in a `DeviceBuffer<Sink>` and be captured by value inside a device lambda, so
 * a single kernel can evaluate a heterogeneous array of boundary sinks. Dispatch is
 * performed by @ref SinkVariant, generated from the `DeviceVariantCase` list below.
 *
 * The engine keeps only the per-particle predicate and the pose advance here; the
 * `parallel_for` over particles and the compaction that actually removes despawned
 * particles are caller concerns and live outside the sink.
 *
 * @note All special members and hot-path methods are `ATLAS_ALL_DEVICE` so the type
 *       is fully usable on the device.
 */
class Sink final {
public:
    /// Active-member discriminator; selects which union leaf is live. Defaults to @ref SinkType::surface.
    SinkType type = SinkType::surface;

    /// Storage for the active leaf; only the member named by @ref type is valid.
    union {

        /// Live when @ref type == @ref SinkType::surface.
        SurfaceSink surface;

        /// Live when @ref type == @ref SinkType::volume.
        VolumeSink volume;

        /// Live when @ref type == @ref SinkType::tracing.
        TracingSink tracing;
    };

    /// Default-constructs an empty surface sink (see the out-of-line definition below).
    ATLAS_ALL_DEVICE
    Sink() noexcept;

    /// Trivial copy: bitwise-copies the tag and active leaf.
    ATLAS_ALL_DEVICE
    Sink(const Sink& other) noexcept = default;

    /// Trivial copy assignment: bitwise-copies the tag and active leaf.
    ATLAS_ALL_DEVICE Sink&
    operator=(const Sink& other) noexcept = default;

    /// Trivial destructor; leaves own no resources requiring cleanup.
    ATLAS_ALL_DEVICE
    ~Sink() noexcept = default;

    /**
     * @brief Constructs a @ref Sink directly from one leaf value.
     *
     * The `enable_if` excludes @ref Sink itself so this does not shadow the copy
     * constructor; the leaf's matching @ref SinkType is deduced by @ref SinkVariant
     * and the union member is placement-constructed from @p op.
     *
     * @tparam Payload A leaf type (@ref SurfaceSink, @ref VolumeSink, @ref TracingSink).
     * @param  op      The leaf value to store; copied into the active union member.
     */
    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Sink>, int> = 0>
    ATLAS_ALL_DEVICE explicit Sink(const Payload& op) noexcept;

    /**
     * @brief Advances the active leaf's boundary unit by one time step.
     * @param dt Time step in seconds; forwarded to the leaf's `advance`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(float dt) noexcept;

    /**
     * @brief Asks the active leaf whether this particle should be removed.
     *
     * Dispatches on @ref type to the leaf's `despawn`. If the tag is somehow
     * unrecognized the visit falls back to `false` (keep the particle).
     *
     * @param position Particle position in world space.
     * @param velocity Particle velocity in world space.
     * @param dt       Time step in seconds.
     * @return `true` if the active leaf votes to despawn, else `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3& velocity, float dt) const noexcept;
};

/**
 * @brief `DeviceVariant` specialization wiring @ref Sink's tag to its union members.
 *
 * Lists one `DeviceVariantCase` per leaf so `construct`, `construct_payload`,
 * `apply`, and `visit` can map a @ref SinkType tag to the correct member pointer.
 * @ref SinkType::surface is the default-constructed case.
 */
using SinkVariant = DeviceVariant<
    Sink,
    SinkType,
    SinkType::surface,
    DeviceVariantCase<SinkType::surface, &Sink::surface>,
    DeviceVariantCase<SinkType::volume, &Sink::volume>,
    DeviceVariantCase<SinkType::tracing, &Sink::tracing>>;

/**
 * @brief Visitor functor that advances whichever leaf is active.
 *
 * Passed to `SinkVariant::apply`; `operator()` is a template so the same functor
 * works for every leaf type. Carries the time step by value.
 */
class SinkAdvance {
public:
    float dt; ///< Time step in seconds forwarded to the leaf's `advance`.
    /**
     * @brief Advances one concrete leaf.
     * @tparam S The active leaf type, deduced by the variant.
     * @param  sink The active leaf, mutated in place.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(S& sink) const noexcept { sink.advance(dt); }
};

/**
 * @brief Visitor functor that evaluates the despawn predicate on the active leaf.
 *
 * Passed to `SinkVariant::visit`; holds the particle state by reference (it lives on
 * the caller's stack for the duration of the visit) and the time step by value.
 */
class SinkDespawn {
public:
    const Float3& position; ///< Particle position in world space.
    const Float3& velocity; ///< Particle velocity in world space.
    float dt;               ///< Time step in seconds.
    /**
     * @brief Runs one concrete leaf's despawn test.
     * @tparam S The active leaf type, deduced by the variant.
     * @param  sink The active leaf.
     * @return The leaf's despawn vote.
     */
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& sink) const noexcept { return sink.despawn(position, velocity, dt); }
};

/// Default constructor: activates the @ref SinkType::surface leaf via the variant.
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Sink::Sink() noexcept {
    SinkVariant::construct(*this, SinkType::surface);
}

/// Payload constructor: deduces the tag from @p op's type and stores it as the active leaf.
template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Sink>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Sink::Sink(const Payload& op) noexcept {
    SinkVariant::construct_payload(*this, op);
}

/// Dispatches @ref advance to the active leaf through @ref SinkAdvance.
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
Sink::advance(const float dt) noexcept {
    SinkVariant::apply(*this, SinkAdvance { dt });
}

/// Dispatches @ref despawn to the active leaf through @ref SinkDespawn; `false` on an unknown tag.
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Sink::despawn(const Float3& position, const Float3& velocity, const float dt) const noexcept {
    return SinkVariant::visit(
        *this,
        SinkDespawn { position, velocity, dt },
        false);
}

/// Host-side shared owner of a @ref Sink.
using SinkHostPtr = atlas::host_shared_ptr<Sink>;

/// Device-side shared owner of a @ref Sink.
using SinkDevicePtr = atlas::device_shared_ptr<Sink>;

}