#pragma once

/**
 * @file codec.h
 * @brief Declares the abstract codec interface and related device-facing helper types.
 *
 * @details
 * This header defines the core codec abstraction used by Atlas to transform
 * simulation state between different representations through encode/decode passes.
 *
 * A codec typically operates at the boundary between:
 * - particle-resolved state carried by @ref atlas::fluid::Fluid,
 * - structured or persistent field data carried by @ref atlas::domain::Domain,
 * - neighborhood/topology information provided by a spatial search structure,
 * - codec-owned device-side scratch or allocation state.
 *
 * The design separates:
 * - **host-side ownership and orchestration**, handled by the polymorphic
 *   @ref atlas::system::Codec interface, and
 * - **device-side execution inputs**, packaged into lightweight probe structures
 *   that may be copied into kernels or backend launch closures.
 *
 * ## Typical lifecycle
 * A codec is commonly used in the following sequence:
 * 1. Construct a concrete codec and associate it with a domain.
 * 2. Create a @ref CodecDeviceProbe via @ref Codec::make_device_probe.
 * 3. Optionally call @ref Codec::update to refresh internal intermediate state.
 * 4. Call @ref Codec::encode to write a coded representation.
 * 5. Call @ref Codec::decode to reconstruct or project state back.
 *
 * ## Probe model
 * To avoid exposing host-only ownership, virtual objects, or heavyweight state
 * inside backend code, codec execution paths operate on POD-like probe objects.
 * These probes contain only the information needed by low-level execution code,
 * such as:
 * - implementation type tags,
 * - raw device pointers,
 * - mutable scratch or allocation flags.
 *
 * ## Ownership and memory
 * The codec owns any persistent buffers required for its operation. In particular,
 * this base class stores a device-side allocation-state buffer used to communicate
 * codec-managed initialization/allocation status to execution code.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout codec-related operations.
 */

#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

/**
 * @brief Identifies the concrete codec implementation category.
 *
 * @details
 * This enum acts as a lightweight runtime type tag for device-side logic and
 * dispatch paths that cannot directly rely on host-side C++ polymorphism.
 *
 * Typical use cases include:
 * - selecting a backend kernel path,
 * - switching implementation-specific logic on device,
 * - serializing or reporting the active codec family.
 */
enum class CodecType : int {
    single,        ///< Single codec implementation.
    knudsen,       ///< Knudsen-based codec implementation.
    deep_learning, ///< Deep learning-based codec implementation.
};

/**
 * @brief Lightweight device-side probe carrying codec execution state.
 *
 * @details
 * This structure is intended to be passed by value into device-executed code.
 * It contains only the minimal codec-specific information needed by kernels,
 * backend lambdas, or other low-level execution paths.
 *
 * Unlike the host-side @ref Codec interface, this probe:
 * - does not own memory,
 * - does not perform polymorphic dispatch,
 * - does not expose host-only resources,
 * - is cheap to copy.
 *
 * A probe instance is typically created by @ref Codec::make_device_probe and then
 * forwarded together with fluid/domain/searcher probes to @ref Codec::update,
 * @ref Codec::encode, and @ref Codec::decode.
 *
 * @tparam T Floating-point scalar type associated with the surrounding codec.
 *
 * @note
 * The template parameter @p T is part of the surrounding codec type system even
 * though this specific probe currently stores no direct scalar-valued members.
 */
template <typename T>
struct CodecDeviceProbe {
    /**
     * @brief Runtime identifier of the concrete codec implementation.
     *
     * @details
     * This value allows device-side logic to distinguish between codec families
     * without relying on host-side virtual dispatch.
     */
    CodecType type;

    /**
     * @brief Pointer to codec-managed device allocation state.
     *
     * @details
     * Points to a device-resident integer flag or state value controlled by the codec.
     * This is typically used to:
     * - record whether codec-side buffers have been initialized,
     * - avoid repeated allocations,
     * - coordinate lazy setup across repeated execution passes.
     *
     * The exact meaning of the pointed-to integer is implementation-defined.
     *
     * @warning
     * This pointer is non-owning. Its lifetime must remain valid for the duration
     * of any device work using the probe.
     */
    int* allocated_system;
};

/**
 * @brief Abstract base class for codec implementations.
 *
 * @details
 * @ref Codec defines the common interface used to transform simulation data
 * through codec-specific encode/decode operations.
 *
 * Concrete derived classes are responsible for implementing:
 * - @ref encode for writing or projecting data into a coded representation,
 * - @ref decode for reconstructing or applying data from that representation,
 * - @ref type for reporting the concrete codec family.
 *
 * The interface is intentionally designed around a mixed host/device execution model:
 * - the object itself is owned and controlled on the host,
 * - execution methods receive lightweight device probes that expose raw data and
 *   backend-friendly state.
 *
 * ## Associated domain
 * A codec may be associated with a @ref Domain object that provides access to:
 * - grid topology,
 * - persistent field storage,
 * - metadata needed to interpret encoded/decoded state.
 *
 * ## Internal state
 * The base class stores:
 * - a host-side domain handle,
 * - a device buffer used to track codec allocation/setup state,
 * - an internal probe counter used to monitor probe refresh activity.
 *
 * ## Expected usage
 * Derived implementations typically follow this pattern:
 * - override @ref update if intermediate structures need refreshing,
 * - allocate or lazily initialize scratch state via @ref codec_probe,
 * - implement encode/decode using the supplied fluid/domain/searcher probes.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Codec {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a codec with default-initialized internal state and no explicitly
     * associated domain handle.
     *
     * @note
     * Concrete implementations may require a valid domain before encode/decode
     * operations are meaningful.
     */
    Codec() = default;

    /**
     * @brief Construct a codec associated with a simulation domain.
     *
     * @details
     * The supplied domain provides codec implementations with access to structured
     * simulation data such as grid-aligned fields and persistent storage used during
     * encoding or decoding.
     *
     * @param domain Host-side shared pointer to the associated simulation domain.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Codec(DomainHostPtr<T> domain);

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Declared virtual to allow correct destruction through base-class pointers.
     */
    virtual ~Codec() = default;

    /**
     * @brief Update codec-internal intermediate state.
     *
     * @details
     * This method provides an optional pre-pass hook that derived codecs may override
     * in order to refresh or rebuild internal state before an encode/decode step.
     *
     * Typical uses include:
     * - lazily allocating device scratch buffers,
     * - refreshing cached coefficients or lookup structures,
     * - synchronizing derived state with the current fluid/domain/searcher state.
     *
     * The default implementation may be a no-op depending on the implementation in
     * `codec.hpp`.
     *
     * @param particle_probe Probe exposing particle-side simulation buffers.
     * @param domain_probe Probe exposing domain/grid fields and metadata.
     * @param searcher_probe Probe exposing spatial hashing / neighbor-search structures.
     * @param codec_probe Mutable codec probe carrying codec-specific device state.
     *
     * @note
     * This function is host-dispatched, but it is designed to prepare data that may
     * later be consumed by device-executed code.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe);

    /**
     * @brief Encode simulation state into a codec-defined representation.
     *
     * @details
     * This pure virtual function must be implemented by all concrete codecs.
     *
     * Conceptually, encoding may include operations such as:
     * - projecting particle data onto grid or latent representations,
     * - compressing or transforming state,
     * - computing auxiliary coded quantities for later reconstruction.
     *
     * The exact semantics are implementation-specific.
     *
     * @param particle_probe Probe exposing particle-side simulation buffers.
     * @param domain_probe Probe exposing domain/grid fields and metadata.
     * @param searcher_probe Probe exposing spatial hashing / neighbor-search structures.
     * @param codec_probe Mutable codec probe carrying codec-specific device state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Decode simulation state from a codec-defined representation.
     *
     * @details
     * This pure virtual function must be implemented by all concrete codecs.
     *
     * Conceptually, decoding may include operations such as:
     * - reconstructing particle or field values from encoded state,
     * - applying inverse transforms,
     * - projecting stored representations back into simulation variables.
     *
     * The exact semantics are implementation-specific.
     *
     * @param particle_probe Probe exposing particle-side simulation buffers.
     * @param domain_probe Probe exposing domain/grid fields and metadata.
     * @param searcher_probe Probe exposing spatial hashing / neighbor-search structures.
     * @param codec_probe Mutable codec probe carrying codec-specific device state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Create a device probe bound to this codec instance.
     *
     * @details
     * Builds and returns a lightweight @ref CodecDeviceProbe referencing the
     * codec's device-visible state.
     *
     * The returned probe is intended to be:
     * - cached by higher-level systems,
     * - copied into backend launch contexts,
     * - reused across update/encode/decode calls as long as the underlying
     *   codec-owned memory remains valid.
     *
     * @return A device-facing probe referencing this codec's runtime state.
     *
     * @note
     * Probe creation typically increments or otherwise interacts with
     * @ref _probe_count to track refresh activity.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Reset codec-internal state.
     *
     * @details
     * Reinitializes internal bookkeeping and/or device-side scratch state so the
     * codec can begin a fresh execution sequence.
     *
     * This is particularly useful when:
     * - restarting a simulation,
     * - invalidating lazy allocations,
     * - forcing codec-side setup to occur again on the next update/encode/decode pass.
     *
     * The exact reset behavior is implementation-defined in `codec.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Return the runtime type tag of the concrete codec.
     *
     * @details
     * This function provides a lightweight categorical identifier for the active
     * codec implementation.
     *
     * @return The corresponding @ref CodecType value for the derived codec.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual CodecType
    type() const noexcept
        = 0;

private:
    /**
     * @brief Internal counter used to track probe creation or refresh activity.
     *
     * @details
     * This counter may be used by the implementation to monitor how often device
     * probes are regenerated or synchronized with host-side state.
     *
     * Its exact semantics are implementation-defined.
     */
    std::uint64_t _probe_count = 0;

    /**
     * @brief Associated simulation domain.
     *
     * @details
     * Stores the host-side domain handle supplied at construction time, if any.
     * Concrete codecs may use this to access persistent grid/field data needed
     * during transformation operations.
     */
    DomainHostPtr<T> _domain {};

    /**
     * @brief Device buffer tracking codec allocation/setup state.
     *
     * @details
     * This buffer backs the @ref CodecDeviceProbe::allocated_system pointer and
     * allows codec logic to communicate persistent initialization state to
     * device-side execution paths.
     *
     * Typical uses include:
     * - lazy allocation guards,
     * - one-time setup flags,
     * - stateful scratch-buffer initialization.
     */
    DeviceBuffer<int> d_allocated_system;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Codec = system::Codec<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

} // namespace atlas

#include <atlas/codec/codec.hpp>