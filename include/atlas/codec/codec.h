#pragma once

#include <atlas/data/particle_data.h>
#include <atlas/domain/domain.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

/**
 * @file codec.h
 * @brief Abstract interface for particle codecs operating on device-side data.
 *
 * @details
 * A **Codec** is responsible for encoding and decoding particle-related information
 * into auxiliary device buffers, typically to support solvers, compression schemes,
 * or domain-specific representations.
 *
 * The design follows the same pattern used throughout Atlas:
 * - A **host-side owning object** (`Codec<T>`)
 * - A lightweight **device-side probe** (`CodecDeviceProbe<T>`) passed into kernels
 *
 * The codec itself does not own particle or domain data; instead, it operates on:
 * - `ParticleDeviceProbe<T>`   : particle state (positions, velocities, etc.)
 * - `DomainDeviceProbe<T>`     : domain discretization and bounds
 * - `SpatialHashingProbe<T>`   : neighbor-search acceleration structure
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @note
 * - `Codec` is an abstract base class; concrete codecs must implement `encode()` and `decode()`.
 * - Device memory used by the codec is owned by the codec instance and exposed via the device probe.
 */

// ------------------------------------------------------------
// Codec types
// ------------------------------------------------------------
/**
 * @brief Enumeration of supported codec types.
 *
 * @details
 * This enum defines the different codec strategies available in the system.
 */
enum class CodecType : int {
    single,
    knudsen,
    deep_learning,
};

// ------------------------------------------------------------
// Device-side probe
// ------------------------------------------------------------

/**
 * @brief Device-side probe for codec-specific buffers.
 *
 * @details
 * This POD-style structure is passed by value into device kernels.
 * It exposes raw device pointers to codec-owned buffers.
 *
 * The exact semantic meaning of the buffers is codec-specific.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct CodecDeviceProbe {
    /// @brief Type of codec.
    CodecType type;

    /// @brief Pointer to a device buffer tracking allocated system indices.
    int* allocated_system;
};

// ------------------------------------------------------------
// Codec interface
// ------------------------------------------------------------

/**
 * @brief Abstract base class for particle codecs.
 *
 * @details
 * A codec transforms particle data to and from a representation suitable for
 * a particular solver, compression strategy, or intermediate computation.
 *
 * Typical lifecycle:
 * 1) Construct codec with a domain
 * 2) Call `encode()` to populate codec-owned buffers from particle/domain/searcher data
 * 3) Use codec buffers in device kernels
 * 4) Call `decode()` to write results back to particle data (if applicable)
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Codec {
public:
    /// @brief Default constructor (domain must be set later if required).
    Codec() = default;

    /**
     * @brief Constructs a codec bound to a specific simulation domain.
     *
     * @param domain Host-side domain shared pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Codec(DomainHostPtr<T> domain);

    /// @brief Virtual destructor.
    virtual ~Codec() = default;

    /**
     * @brief Updates internal codec state based on current particle and domain data.
     *
     * @details
     * This function allows the codec to refresh or recompute any internal
     * buffers or metadata before encoding/decoding operations.
     *
     * @param particle_probe Device probe exposing particle data.
     * @param domain_probe   Device probe exposing domain metadata.
     * @param searcher_probe Device probe for spatial hashing neighbor queries.
     * @param codec_probe    Device probe exposing codec-owned buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe);

    /**
     * @brief Encodes particle data into codec-specific device buffers.
     *
     * @details
     * This function is expected to:
     * - Inspect particle data
     * - Potentially use domain and neighbor-search information
     * - Populate or update buffers referenced by `codec_probe`
     *
     * @param particle_probe Device probe exposing particle data.
     * @param domain_probe   Device probe exposing domain metadata.
     * @param searcher_probe Device probe for spatial hashing neighbor queries.
     * @param codec_probe    Device probe exposing codec-owned buffers.
     *
     * @note
     * - This is a host-entry function.
     * - Implementations may launch device kernels internally.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Decodes codec-specific data back into particle state.
     *
     * @details
     * This function performs the inverse (or complementary) operation of `encode()`,
     * typically writing results back to particle buffers.
     *
     * @param particle_probe Device probe exposing particle data.
     * @param domain_probe   Device probe exposing domain metadata.
     * @param searcher_probe Device probe for spatial hashing neighbor queries.
     * @param codec_probe    Device probe exposing codec-owned buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    /**
     * @brief Produces a device-side probe exposing codec buffers.
     *
     * @return `CodecDeviceProbe<T>` containing raw device pointers.
     *
     * @note
     * - The returned probe does not own memory.
     * - Pointer validity is guaranteed until the codec is destroyed or `reset()` is called.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Resets internal buffers and invalidates device probes.
     *
     * @details
     * After calling `reset()`, any previously obtained `CodecDeviceProbe`
     * becomes invalid and `make_device_probe()` must be called again.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Returns the codec type.
     *
     * @return `CodecType` enumeration value identifying the codec strategy.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual CodecType
    type() const noexcept = 0;

private:
    /// @brief Counter used to track probe creation or versioning (implementation-defined).
    std::uint64_t _probe_count = 0;

    /// @brief Domain associated with this codec.
    DomainHostPtr<T> _domain {};

    /// @brief Device buffer storing allocated system indices (codec-specific semantics).
    DeviceBuffer<int> d_allocated_system;
};

} // namespace atlas::system

// ------------------------------------------------------------
// Public aliases
// ------------------------------------------------------------

namespace atlas {

/**
 * @brief Public alias for `system::Codec<T>`.
 */
template <typename T>
using Codec = system::Codec<T>;

/**
 * @brief Host-side shared pointer to a codec.
 */
template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

/**
 * @brief Device-side shared pointer to a codec.
 */
template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

} // namespace atlas

// ------------------------------------------------------------
// Inline / implementation
// ------------------------------------------------------------

#include <atlas/codec/codec.hpp>
