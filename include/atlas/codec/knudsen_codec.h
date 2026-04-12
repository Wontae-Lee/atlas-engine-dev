#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Codec that evaluates a Knudsen-style field over the simulation domain.
 *
 * KnudsenCodec augments the generic codec abstraction with a characteristic
 * length scale and a device buffer that stores per-cell Knudsen-related values.
 * This is useful when the simulation must switch behavior based on local
 * rarefaction or another nondimensional criterion tied to cell state.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for KnudsenCodec.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    KnudsenCodec() = default;

    /**
     * @brief Constructs the codec with its domain and reference length scale.
     *
     * @param domain Domain associated with this codec instance.
     * @param characteristic_length Reference length used in Knudsen evaluation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length);

    /**
     * @brief Virtual destructor.
     */
    ~KnudsenCodec() override = default;

    /**
     * @brief Encodes simulation state into per-cell Knudsen data.
     *
     * The codec can inspect particle, field, and neighborhood information to
     * populate or refresh the internal Knudsen buffer.
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
     * @brief Decodes Knudsen-derived information back into runtime state.
     *
     * This hook allows downstream stages to consume the codec's reduced field
     * representation and reconstruct the state they need.
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
     * @return CodecType::knudsen
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
    ///< Reference length scale used when computing Knudsen values.
    T _characteristic_length = T(1);

    ///< Device-resident storage for the codec's per-cell Knudsen field.
    DeviceBuffer<T> d_knudsen_values {};
};

template <typename T>
class KnudsenCodec<T>::Builder final {
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
     * @brief Sets the characteristic length for Knudsen evaluation.
     *
     * @param characteristic_length Reference physical length scale.
     * @return Builder& Fluent reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    /**
     * @brief Builds a value instance after validation.
     *
     * @return KnudsenCodec<T> Fully initialized codec value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    /**
     * @brief Builds a host-shared codec instance after validation.
     *
     * @return atlas::host_shared_ptr<KnudsenCodec<T>> Shared codec object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
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

    ///< Characteristic length stored until construction time.
    T _characteristic_length = T(1);
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::KnudsenCodec.
 */
template <typename T>
using KnudsenCodec = system::KnudsenCodec<T>;

/**
 * @brief Host shared pointer alias for KnudsenCodec.
 */
template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<system::KnudsenCodec<T>>;

/**
 * @brief Device shared pointer alias for KnudsenCodec.
 */
template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<system::KnudsenCodec<T>>;

}

#include <atlas/codec/knudsen_codec.hpp>
