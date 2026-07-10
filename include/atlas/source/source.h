#pragma once

#include <atlas/core/host_variant.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/memory.h>
#include <atlas/source/source_type.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>

#include <concepts>
#include <cstddef>
#include <utility>

namespace atlas {

/**
 * @brief Compile-time contract every `Source` leaf must satisfy.
 *
 * A conforming leaf can emit particles into a fluid position buffer and advance
 * its own boundary in time. `spawn` is queried on a const leaf because emission
 * must not mutate the leaf (its cache is fixed); `advance` is non-const because it
 * moves the underlying unit.
 *
 * @tparam S Candidate leaf type.
 */
template <typename S>
concept ConceptSource = requires(S source, const S const_source, FluidPositionState* positions, std::size_t offset, float dt) {
    { const_source.spawn(positions, offset) } -> std::same_as<int>;
    { source.advance(dt) } -> std::same_as<void>;
};

static_assert(ConceptSource<SurfaceSource>);
static_assert(ConceptSource<VolumeSource>);

/**
 * @brief Tagged-union umbrella over the particle-emission leaves.
 *
 * `Source` holds exactly one leaf — a `SurfaceSource` or a `VolumeSource` —
 * selected by @ref type, and forwards `spawn`/`advance` to the active leaf. All
 * union bookkeeping (which member is alive, how to construct/move/destroy it) is
 * delegated to `SourceVariant`, a `HostVariant`.
 *
 * It is a **`HostVariant`, not a `DeviceVariant`**, because each leaf owns a
 * `DeviceBuffer<Float3>` cache whose copy and destructor are host-only: a
 * `DeviceVariant`'s `__host__ __device__` union machinery would try to instantiate
 * those host-only members for the device and fail to compile (nvcc #20014). A
 * `HostVariant` keeps the union bookkeeping host-only. Consequently `Source` is
 * host-side and **move-only** (copy is deleted).
 *
 * @note The data members are public because the `HostVariant` free functions
 *       operate on them directly; this is not an invitation to poke at the union
 *       without going through the member functions.
 * @see SourceVariant, ConceptSource, System::emit
 */
class Source final {
public:
    /// Active-member discriminant; selects which union leaf is alive.
    SourceType type = SourceType::surface;

    /**
     * @brief Storage for the active leaf; exactly one member is alive at a time.
     *
     * The alive member is the one named by @ref type. It is constructed, moved,
     * and destroyed only through `SourceVariant`, never by the compiler's implicit
     * union rules (which are deleted for non-trivial members).
     */
    union {

        /// Alive when `type == SourceType::surface`.
        SurfaceSource surface;

        /// Alive when `type == SourceType::volume`.
        VolumeSource volume;
    };

    /**
     * @brief Constructs a source holding a default-constructed `SurfaceSource`.
     *
     * The default leaf has an empty cache, so this source emits nothing until it is
     * assigned a built leaf.
     */
    ATLAS_HOST
    Source() noexcept;

    /**
     * @brief Constructs a source that owns the given `SurfaceSource` leaf.
     * @param op Leaf to adopt; moved into the union and sets `type` to `surface`.
     */
    ATLAS_HOST explicit Source(SurfaceSource op) noexcept;

    /**
     * @brief Constructs a source that owns the given `VolumeSource` leaf.
     * @param op Leaf to adopt; moved into the union and sets `type` to `volume`.
     */
    ATLAS_HOST explicit Source(VolumeSource op) noexcept;

    /// Deleted: leaves own device buffers with host-only, non-copyable semantics.
    Source(const Source&) = delete;

    /// Deleted: leaves own device buffers with host-only, non-copyable semantics.
    Source&
    operator=(const Source&)
        = delete;

    /**
     * @brief Move-constructs from @p other, adopting its active leaf.
     * @param other Source to move from; left in a valid but unspecified state.
     */
    ATLAS_HOST
    Source(Source&& other) noexcept;

    /**
     * @brief Move-assigns from @p other, replacing the current active leaf.
     * @param other Source to move from; left in a valid but unspecified state.
     * @return `*this`.
     */
    ATLAS_HOST Source&
    operator=(Source&& other) noexcept;

    /// Destroys the currently active leaf through `SourceVariant`.
    ATLAS_HOST
    ~Source() noexcept;

    /**
     * @brief Advances the active leaf's boundary by @p dt seconds.
     * @param dt Timestep in seconds; forwarded to the active leaf's `advance`.
     */
    ATLAS_HOST void
    advance(float dt) noexcept;

    /**
     * @brief Emits the active leaf's cached points into a fluid position buffer.
     *
     * Forwards to the active leaf's `spawn`. If `type` somehow names no live case,
     * the `SourceVariant::visit` fallback returns 0.
     *
     * @param positions Destination fluid position state; null writes nothing.
     * @param offset    Index of the first slot to write (the current particle count).
     * @return Number of particles written by the active leaf (0 if none).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    spawn(FluidPositionState* positions, std::size_t offset) const;
};

/**
 * @brief `HostVariant` specialization that drives `Source`'s union bookkeeping.
 *
 * Binds each `SourceType` tag to the matching union member pointer so the generic
 * `HostVariant` machinery can construct, move, destroy, and dispatch on the active
 * leaf. `SourceType::surface` is the default tag used when a requested tag is
 * unknown.
 */
using SourceVariant = HostVariant<
    Source,
    SourceType,
    SourceType::surface,
    HostVariantCase<SourceType::surface, &Source::surface>,
    HostVariantCase<SourceType::volume, &Source::volume>>;

/**
 * @brief Visitor that advances whichever leaf is active by a stored timestep.
 *
 * Used with `SourceVariant::apply`; the templated call operator matches any leaf.
 */
class SourceAdvance {
public:
    float dt; ///< Timestep in seconds forwarded to the active leaf.

    /**
     * @brief Advances the given leaf.
     * @tparam S Active leaf type deduced by the variant dispatch.
     * @param source Mutable reference to the active leaf.
     */
    template <typename S>
    ATLAS_HOST void
    operator()(S& source) const noexcept { source.advance(dt); }
};

/**
 * @brief Visitor that spawns from whichever leaf is active into a stored target.
 *
 * Used with `SourceVariant::visit`; the templated call operator matches any leaf
 * and returns the leaf's spawn count.
 */
class SourceSpawn {
public:
    FluidPositionState* positions; ///< Destination fluid position state.
    std::size_t offset;            ///< Index of the first slot to write.

    /**
     * @brief Spawns from the given leaf into the stored target.
     * @tparam S Active leaf type deduced by the variant dispatch.
     * @param source Const reference to the active leaf.
     * @return Number of particles the leaf wrote.
     */
    template <typename S>
    ATLAS_HOST int
    operator()(const S& source) const { return source.spawn(positions, offset); }
};

// Default-constructs the surface leaf as the initially-active member.
ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source() noexcept {
    SourceVariant::construct(*this, SourceType::surface);
}

// Adopts a SurfaceSource payload; construct_payload deduces the matching tag.
ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(SurfaceSource op) noexcept {
    SourceVariant::construct_payload(*this, std::move(op));
}

// Adopts a VolumeSource payload; construct_payload deduces the matching tag.
ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(VolumeSource op) noexcept {
    SourceVariant::construct_payload(*this, std::move(op));
}

// Move-constructs the active leaf of `other` into this fresh union.
ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(Source&& other) noexcept {
    SourceVariant::move_construct(*this, std::move(other));
}

// Tears down the current leaf if needed and move-adopts `other`'s active leaf.
ATLAS_HOST ATLAS_FORCE_INLINE Source&
Source::operator=(Source&& other) noexcept {
    SourceVariant::move_assign(*this, std::move(other));
    return *this;
}

// Destroys whichever union member is currently alive.
ATLAS_HOST ATLAS_FORCE_INLINE
    Source::~Source() noexcept {
    SourceVariant::destroy(*this);
}

// Dispatches advance() to the active leaf via the apply visitor.
ATLAS_HOST ATLAS_FORCE_INLINE void
Source::advance(const float dt) noexcept {
    SourceVariant::apply(*this, SourceAdvance { dt });
}

// Dispatches spawn() to the active leaf; the trailing 0 is the visit fallback
// returned when `type` matches no registered case.
ATLAS_HOST ATLAS_FORCE_INLINE int
Source::spawn(FluidPositionState* positions, const std::size_t offset) const {
    return SourceVariant::visit(*this, SourceSpawn { positions, offset }, 0);
}

/// Host-side shared-ownership pointer to a `Source` umbrella.
using SourceHostPtr = atlas::host_shared_ptr<Source>;

/// Device-side shared-ownership pointer alias for a `Source` umbrella.
using SourceDevicePtr = atlas::device_shared_ptr<Source>;

}