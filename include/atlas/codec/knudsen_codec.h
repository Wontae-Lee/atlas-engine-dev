#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Codec that computes and stores per-cell Knudsen numbers (rarefaction metric).
 *
 * @details
 * `KnudsenCodec` is a domain-aware codec that derives a Knudsen-related quantity
 * from the particle state and neighborhood structure and stores it on the device.
 *
 * In many continuum/rarefied flow models, the Knudsen number is defined as:
 * \f[
 *   \mathrm{Kn} = \frac{\lambda}{L}
 * \f]
 * where:
 * - \f$\lambda\f$ is a microscopic length scale (e.g., mean free path),
 * - \f$L\f$ is a macroscopic characteristic length.
 *
 * This codec takes \f$L\f$ as `characteristic_length` and stores it as
 * `_characteristic_length`. The actual definition/estimation of \f$\lambda\f$
 * depends on your implementation (e.g., local number density from neighbor counts,
 * collision model parameters, etc.) and is performed inside `encode()`/`decode()`.
 *
 * @par Typical usage pattern
 * - `encode()`: accumulate local statistics and compute a per-cell Knudsen value,
 *   writing results into `d_knudsen_values`.
 * - `decode()`: optionally apply these values to domain fields or drive model
 *   selection (e.g., continuum vs. rarefied solver allocation).
 *
 * @par Device storage
 * The codec owns a device buffer:
 * - `d_knudsen_values`: usually sized to the number of domain cells (one value per cell).
 *
 * @par Probes
 * The probes passed to `encode()`/`decode()` are non-owning views:
 * - `ParticleDeviceProbe<T>`: particle attributes on device (positions, alive count, etc.)
 * - `DomainDeviceProbe<T>`: domain discretization and device-side domain fields
 * - `SpatialHashingProbe<T>`: neighbor mapping (cell ranges and particle indices)
 * - `CodecDeviceProbe<T>`: codec-related device pointers exposed to kernels (if used)
 *
 * @tparam T Floating-point scalar type (typically `float` or `double`).
 *
 * @see Codec
 * @see Domain
 * @see SpatialHashingProbe
 */
template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    /// @brief Fluent builder for configuring and constructing a `KnudsenCodec`.
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an unbound codec instance. A valid domain and characteristic length
     * must be provided (e.g., via the explicit constructor or builder) before the
     * codec is used in a simulation pipeline.
     */
    KnudsenCodec() = default;

    /**
     * @brief Construct a Knudsen codec bound to a domain and characteristic length.
     *
     * @details
     * The `domain` argument is required by the base class `Codec<T>` for
     * domain-dependent initialization (e.g., allocating per-cell buffers).
     * The `characteristic_length` provides the macroscopic length scale \f$L\f$
     * used to compute the Knudsen number.
     *
     * Implementations typically resize `d_knudsen_values` to the domain cell count
     * (e.g., `domain->number_of_cells()`) during construction or at first use.
     *
     * @param domain Host pointer to the domain object (must be non-null).
     * @param characteristic_length Positive characteristic length \f$L\f$.
     *
     * @pre `domain != nullptr`
     * @pre `characteristic_length > 0`
     *
     * @throws std::invalid_argument
     * If `domain` is null (usually validated by the base `Codec<T>`) or if the
     * characteristic length is invalid (builder/implementation dependent).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length);

    /// @brief Virtual destructor.
    ~KnudsenCodec() override = default;

    /**
     * @brief Encode the current state into a per-cell Knudsen field.
     *
     * @details
     * This function is intended to compute and store Knudsen-related values
     * based on the current particle distribution and neighborhood information.
     *
     * A typical implementation may:
     * - use `searcher_probe` to iterate particles per cell and/or neighbors,
     * - estimate a local microscopic scale (e.g., from density or collision model),
     * - compute \f$\mathrm{Kn} = \lambda / L\f$ using `_characteristic_length`,
     * - write the result into `d_knudsen_values` (often one value per domain cell).
     *
     * Depending on how `CodecDeviceProbe<T>` is designed, the codec may also
     * expose a device pointer to the computed field via `codec_probe`.
     *
     * @param particle_probe Particle data probe (device view, non-owning).
     * @param domain_probe   Domain data probe (device view, non-owning).
     * @param searcher_probe Spatial hashing probe for neighbor/cell traversal (device view).
     * @param codec_probe    Codec probe for device-side codec state exposure (optional usage).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decode/apply the stored Knudsen field to downstream buffers or logic.
     *
     * @details
     * The decode step is typically used to apply or export the per-cell Knudsen
     * values computed in `encode()`. Examples include:
     * - writing/accumulating values into domain fields (e.g., temperature, flags),
     * - selecting between solver models per cell (continuum vs. rarefied),
     * - producing diagnostics or thresholds used by later kernels.
     *
     * The exact behavior is implementation-specific; if no application is needed,
     * this function may be a no-op.
     *
     * @param particle_probe Particle data probe (device view, non-owning).
     * @param domain_probe   Domain data probe (device view, non-owning).
     * @param searcher_probe Spatial hashing probe (device view, non-owning).
     * @param codec_probe    Codec probe for device-side codec state exposure (optional usage).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Builder entry point for fluent construction.
     *
     * @return A default-initialized `Builder`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return the runtime codec tag for this Knudsen-based implementation.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
    /// @brief Macroscopic characteristic length \f$L\f$ used in Knudsen computation.
    T _characteristic_length = T(1);

    /**
     * @brief Device buffer storing Knudsen-related values.
     *
     * @details
     * Typically sized to `num_of_cells` and interpreted as one value per grid cell.
     * Ownership remains within the codec.
     */
    DeviceBuffer<T> d_knudsen_values {};
};

/**
 * @brief Fluent builder for `KnudsenCodec<T>`.
 *
 * @details
 * The builder gathers construction parameters, validates them, and constructs:
 * - a codec by value via `build()`, or
 * - a host shared pointer via `make_host_shared()`.
 *
 * Required parameters:
 * - `with_domain()` must be called with a non-null domain.
 * - `with_characteristic_length()` should be set to a strictly positive value.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    /// @brief Construct an empty builder.
    Builder() = default;

    /**
     * @brief Set the domain used to size and interpret per-cell storage.
     *
     * @param domain Host pointer to a domain object (must be non-null).
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Set the characteristic length \f$L\f$ used for Knudsen computation.
     *
     * @param characteristic_length Positive macroscopic length scale.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    /**
     * @brief Build a configured codec (returned by value).
     *
     * @return A fully constructed `KnudsenCodec<T>`.
     *
     * @throws std::invalid_argument If the builder is missing required parameters
     *         or if the configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    /**
     * @brief Build a configured codec as a host-shared pointer.
     *
     * @return `host_shared_ptr<KnudsenCodec<T>>` owning the codec.
     *
     * @throws std::invalid_argument If the builder is missing required parameters
     *         or if the configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate builder state and throw on invalid configuration.
     *
     * @details
     * Typical validation includes:
     * - `_domain` must not be null,
     * - `_characteristic_length` must be strictly positive.
     *
     * @throws std::invalid_argument On invalid state.
     */
    void
    validate() const;

private:
    /// @brief Domain used by the codec (required).
    DomainHostPtr<T> _domain {};

    /// @brief Characteristic length \f$L\f$ (must be > 0).
    T _characteristic_length = T(1);
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using KnudsenCodec = system::KnudsenCodec<T>;

template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<system::KnudsenCodec<T>>;

template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<system::KnudsenCodec<T>>;

} // namespace atlas

// Template implementation header.
#include <atlas/codec/knudsen_codec.hpp>
