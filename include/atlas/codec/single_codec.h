#pragma once

#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Minimal codec that keeps the simulation in a single resolved state.
 *
 * SingleCodec is the simplest concrete codec implementation in Atlas. It is
 * suitable when the simulation pipeline requires a codec object for uniform
 * orchestration, but no reduced representation or alternate decoding path is
 * needed.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class SingleCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for SingleCodec.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    SingleCodec() = default;

    /**
     * @brief Constructs the codec for a specific domain.
     *
     * @param domain Domain associated with this codec instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SingleCodec(const DomainHostPtr<T>& domain);

    /**
     * @brief Virtual destructor.
     */
    ~SingleCodec() override = default;

    /**
     * @brief Encodes the current simulation state.
     *
     * For SingleCodec, encode() preserves the direct one-to-one simulation
     * representation rather than projecting into a reduced state.
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
     * @brief Decodes codec-managed data back into direct simulation state.
     *
     * In the single-representation case, decode() serves as the symmetric hook
     * expected by the codec abstraction even when little or no transformation is
     * required.
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
     * @return CodecType::single
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;
};

template <typename T>
class SingleCodec<T>::Builder final {
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
     * @return SingleCodec<T> Fully initialized codec value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SingleCodec<T>
    build() const;

    /**
     * @brief Builds a host-shared codec instance after validation.
     *
     * @return atlas::host_shared_ptr<SingleCodec<T>> Shared codec object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SingleCodec<T>>
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
 * @brief Convenience alias for atlas::system::SingleCodec.
 */
template <typename T>
using SingleCodec = system::SingleCodec<T>;

/**
 * @brief Host shared pointer alias for SingleCodec.
 */
template <typename T>
using SingleCodecHostPtr = atlas::host_shared_ptr<system::SingleCodec<T>>;

/**
 * @brief Device shared pointer alias for SingleCodec.
 */
template <typename T>
using SingleCodecDevicePtr = atlas::device_shared_ptr<system::SingleCodec<T>>;

}

#include <atlas/codec/single_codec.hpp>
