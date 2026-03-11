#pragma once

#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Placeholder codec interface for ML-driven encoding/decoding pipelines.
 *
 * @details
 * `DeepLearningCodec` is a `Codec<T>` implementation intended to integrate
 * deep-learning inference (and possibly feature extraction) into the simulation
 * pipeline. The class currently defines the API surface and builder pattern, but
 * the internal model storage, feature buffers, and inference execution are not yet
 * implemented.
 *
 * Conceptually, this codec can be used to:
 * - **Encode**: transform particle/domain/neighborhood state into ML-friendly
 *   feature tensors (e.g., per-cell feature grids, per-particle features, or
 *   neighborhood-aggregated statistics).
 * - **Decode**: apply ML outputs back to simulation state (e.g., corrective forces,
 *   closure terms, viscosity models, solver allocation masks, etc.).
 *
 * The codec follows Atlas conventions:
 * - The codec is **domain-bound** (constructed with a non-null `DomainHostPtr<T>`),
 *   enabling per-cell allocations based on the domain grid resolution.
 * - `encode()` / `decode()` accept non-owning probes for particles, domain,
 *   and neighbor traversal (`SpatialHashingProbe<T>`).
 *
 * @note
 * Because the implementation is not provided yet, this class should be treated as
 * a **stub**. Calling `encode()`/`decode()` may currently be a no-op depending on
 * the implementation in `deep_learning_codec.hpp`.
 *
 * @tparam T Floating-point scalar type (typically `float` or `double`).
 *
 * @see Codec
 * @see Domain
 * @see SpatialHashingProbe
 */
template <typename T>
class DeepLearningCodec final : public Codec<T> {
public:
    /// @brief Fluent builder for configuring and constructing the codec.
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an unbound codec instance. A domain must be supplied via the
     * explicit constructor or the builder before use.
     */
    DeepLearningCodec() = default;

    /**
     * @brief Construct a deep-learning codec bound to a domain.
     *
     * @details
     * The `DomainHostPtr<T>` is required by the base `Codec<T>` to initialize
     * domain-dependent internal state (e.g., per-cell buffers sized to the domain).
     *
     * Future implementations typically use this domain binding to:
     * - allocate per-cell feature grids / output grids,
     * - store normalization constants derived from domain geometry,
     * - precompute mappings between particles and cells for feature aggregation.
     *
     * @param domain Host pointer to a domain object (must be non-null).
     *
     * @pre `domain != nullptr`
     *
     * @throws std::invalid_argument
     * If `domain` is null (usually validated by the base `Codec<T>`).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit DeepLearningCodec(const DomainHostPtr<T>& domain);

    /// @brief Virtual destructor.
    ~DeepLearningCodec() override = default;

    /**
     * @brief Encode simulation state into ML features and/or intermediate buffers.
     *
     * @details
     * Intended responsibilities (implementation-specific):
     * - Aggregate particle attributes into per-cell or per-particle feature vectors.
     * - Use `searcher_probe` to compute neighborhood-based statistics efficiently.
     * - Write features into codec-owned device buffers or expose pointers through
     *   `codec_probe` for downstream kernels/inference code.
     *
     * Typical feature categories might include:
     * - local number density / occupancy,
     * - velocity moments,
     * - temperature/pressure estimates,
     * - gradients estimated from neighborhood samples.
     *
     * @param particle_probe Particle data probe (device view, non-owning).
     * @param domain_probe   Domain data probe (device view, non-owning).
     * @param searcher_probe Spatial hashing probe for neighbor/cell traversal.
     * @param codec_probe    Codec probe for device-side codec state exposure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decode ML outputs back into simulation state.
     *
     * @details
     * Intended responsibilities (implementation-specific):
     * - Run or consume ML inference outputs.
     * - Apply results to domain fields or particle attributes.
     * - Optionally write masks/flags (e.g., solver selection per cell) into
     *   codec/domain buffers.
     *
     * Examples of decode outputs:
     * - corrective accelerations / forces,
     * - closure terms for subgrid models,
     * - per-cell material/phase classification,
     * - adaptive solver allocation signals.
     *
     * @param particle_probe Particle data probe (device view, non-owning).
     * @param domain_probe   Domain data probe (device view, non-owning).
     * @param searcher_probe Spatial hashing probe for neighbor/cell traversal.
     * @param codec_probe    Codec probe for device-side codec state exposure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
    /**
     * @note
     * DeepLearningCodec-specific members are intentionally not defined yet.
     *
     * Future members commonly include:
     * - model handle / inference runtime (TensorRT, ONNX Runtime, etc.),
     * - device buffers for input features and output predictions,
     * - normalization parameters,
     * - stream/event handles for async execution.
     */
    // TODO: Add any DeepLearningCodec-specific members here.
    // TODO: Not implemented yet.
};

/**
 * @brief Fluent builder for @ref DeepLearningCodec.
 *
 * @details
 * The builder configures required parameters, validates them, and constructs either:
 * - a codec by value (`build()`), or
 * - a shared host instance (`make_host_shared()`).
 *
 * Required parameters:
 * - `with_domain()` must be called with a non-null domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class DeepLearningCodec<T>::Builder final {
public:
    /// @brief Construct an empty builder.
    Builder() = default;

    /**
     * @brief Set the domain required by the base @ref Codec.
     *
     * @param domain Host pointer to a domain object (must be non-null).
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Build a configured codec (returned by value).
     *
     * @return A fully constructed `DeepLearningCodec<T>`.
     *
     * @throws std::invalid_argument If required configuration is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeepLearningCodec<T>
    build() const;

    /**
     * @brief Build a configured codec as a host-shared pointer.
     *
     * @return `host_shared_ptr<DeepLearningCodec<T>>` owning the codec.
     *
     * @throws std::invalid_argument If required configuration is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DeepLearningCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate builder state and throw on invalid configuration.
     *
     * @details
     * Typical validation includes:
     * - `_domain` must not be null.
     *
     * @throws std::invalid_argument On invalid builder state.
     */
    void
    validate() const;

private:
    /// @brief Domain required to construct the codec (must be non-null).
    DomainHostPtr<T> _domain {};
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DeepLearningCodec = system::DeepLearningCodec<T>;

template <typename T>
using DeepLearningCodecHostPtr = atlas::host_shared_ptr<system::DeepLearningCodec<T>>;

template <typename T>
using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<system::DeepLearningCodec<T>>;

} // namespace atlas

#include <atlas/codec/deep_learning_codec.hpp>
