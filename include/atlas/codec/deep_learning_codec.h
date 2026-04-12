#pragma once

#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Placeholder codec for deep-learning-driven compression or reconstruction.
 *
 * This codec exists to reserve a concrete codec type for workflows where
 * encode()/decode() may eventually delegate to a neural network or another
 * learned surrogate model. The current interface mirrors the rest of the codec
 * family so higher-level systems can switch codec implementations without
 * changing orchestration code.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class DeepLearningCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for DeepLearningCodec.
     *
     * The builder captures required construction inputs, validates them, and
     * then materializes either a stack object or a host-shared pointer.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    DeepLearningCodec() = default;

    /**
     * @brief Constructs the codec for a specific simulation domain.
     *
     * @param domain Domain used to size or contextualize codec-side state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit DeepLearningCodec(const DomainHostPtr<T>& domain);

    /**
     * @brief Virtual destructor.
     */
    ~DeepLearningCodec() override = default;

    /**
     * @brief Encodes the current simulation state into codec-managed data.
     *
     * The exact representation is implementation-defined. For this codec type,
     * the method marks the interface point where a learned encoder would
     * inspect particles, domain fields, and neighborhood information.
     *
     * @param particle_probe Active fluid state on the selected backend.
     * @param domain_probe Domain field data and grid metadata.
     * @param searcher_probe Spatial neighborhood indexing state.
     * @param codec_probe Mutable codec-side device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decodes codec-managed data back into simulation state.
     *
     * This is the symmetric entry point to encode(). A future implementation
     * can reconstruct particle or field quantities from compressed latent data.
     *
     * @param particle_probe Active fluid state on the selected backend.
     * @param domain_probe Domain field data and grid metadata.
     * @param searcher_probe Spatial neighborhood indexing state.
     * @param codec_probe Mutable codec-side device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Creates a builder configured with default values.
     *
     * @return Builder Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Returns the runtime codec identifier.
     *
     * @return CodecType::deep_learning
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
};

template <typename T>
class DeepLearningCodec<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Binds the domain required by the codec instance.
     *
     * @param domain Host-side domain pointer used during construction.
     * @return Builder& Fluent reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Builds a value instance after validation.
     *
     * @return DeepLearningCodec<T> Fully initialized codec value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeepLearningCodec<T>
    build() const;

    /**
     * @brief Builds a host-shared codec instance after validation.
     *
     * @return atlas::host_shared_ptr<DeepLearningCodec<T>> Shared codec object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DeepLearningCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates builder state before construction.
     */
    void
    validate() const;

private:
    ///< Domain dependency forwarded into the final codec instance.
    DomainHostPtr<T> _domain {};
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::DeepLearningCodec.
 */
template <typename T>
using DeepLearningCodec = system::DeepLearningCodec<T>;

/**
 * @brief Host shared pointer alias for DeepLearningCodec.
 */
template <typename T>
using DeepLearningCodecHostPtr = atlas::host_shared_ptr<system::DeepLearningCodec<T>>;

/**
 * @brief Device shared pointer alias for DeepLearningCodec.
 */
template <typename T>
using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<system::DeepLearningCodec<T>>;

}

#include <atlas/codec/deep_learning_codec.hpp>
