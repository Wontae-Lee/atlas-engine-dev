#pragma once

/**
 * @file codec.h
 * @brief Declares the base codec interface and related types.
 *
 * This header defines the core codec abstraction used to perform
 * encode/decode operations on simulation data.
 *
 * The codec operates on device-side probes and interacts with
 * fluid, domain, and spatial search structures.
 */

#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

/**
 * @brief Identifies the concrete codec implementation type.
 */
enum class CodecType : int {
    single,        ///< Single codec implementation
    knudsen,       ///< Knudsen-based codec implementation
    deep_learning, ///< Deep learning-based codec implementation
};

/**
 * @brief Device-side probe structure for codec state.
 *
 * This structure is passed to device-executed functions and contains
 * pointers and state required by codec implementations. The probe is designed
 * to be a lightweight transport object that can be copied into kernels or
 * backend-specific lambdas without bringing along host-only ownership.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct CodecDeviceProbe {

    CodecType type; ///< Codec type identifier.

    int* allocated_system; ///< Pointer to codec-controlled device allocation state.
};

/**
 * @brief Abstract base class for codec implementations.
 *
 * Defines the interface for encoding and decoding operations
 * performed on simulation data.
 *
 * Concrete implementations must override encode(), decode(),
 * and type().
 *
 * @tparam T Scalar type.
 */
template <typename T>
class Codec {
public:
    /**
     * @brief Default constructor.
     */
    Codec() = default;

    /**
     * @brief Constructs a codec with an associated domain.
     *
     * The domain gives codec implementations access to the simulation grid and
     * any persistent field storage they need to interpret or transform state.
     *
     * @param domain Host-side domain pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Codec(DomainHostPtr<T> domain);

    /**
     * @brief Virtual destructor.
     */
    virtual ~Codec() = default;

    /**
     * @brief Updates internal codec state.
     *
     * This function may be overridden to update intermediate state
     * before encoding or decoding.
     *
     * @param particle_probe Fluid probe describing particle-side buffers.
     * @param domain_probe Domain probe describing fields and grid metadata.
     * @param searcher_probe Spatial hashing probe with neighborhood indexing.
     * @param codec_probe Codec probe receiving mutable codec-side state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe);

    /**
     * @brief Encodes simulation data.
     *
     * Must be implemented by derived classes.
     *
     * @param particle_probe Fluid probe describing particle-side buffers.
     * @param domain_probe Domain probe describing fields and grid metadata.
     * @param searcher_probe Spatial hashing probe with neighborhood indexing.
     * @param codec_probe Codec probe receiving mutable codec-side state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Decodes simulation data.
     *
     * Must be implemented by derived classes.
     *
     * @param particle_probe Fluid probe describing particle-side buffers.
     * @param domain_probe Domain probe describing fields and grid metadata.
     * @param searcher_probe Spatial hashing probe with neighborhood indexing.
     * @param codec_probe Codec probe receiving mutable codec-side state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Creates a device probe for this codec.
     *
     * The returned probe is intended to be cached by higher-level systems and
     * passed into update(), encode(), and decode() without exposing codec
     * ownership details to backend code.
     *
     * @return CodecDeviceProbe<T> Device-side probe structure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Resets internal state.
     *
     * Implementations that keep device-side scratch or allocation tracking can
     * use this to reinitialize their internal state before a new simulation run.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Returns the codec type.
     *
     * @return CodecType Identifier of the concrete implementation.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual CodecType
    type() const noexcept = 0;

private:
    std::uint64_t _probe_count = 0; ///< Internal counter used to track probe refreshes.

    DomainHostPtr<T> _domain {}; ///< Associated simulation domain.

    DeviceBuffer<int> d_allocated_system; ///< Device buffer used to track codec allocation state.
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for system::Codec.
 */
template <typename T>
using Codec = system::Codec<T>;

/**
 * @brief Host shared pointer to Codec.
 */
template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

/**
 * @brief Device shared pointer to Codec.
 */
template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

} // namespace atlas

#include <atlas/codec/codec.hpp>
