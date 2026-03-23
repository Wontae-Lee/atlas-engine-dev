#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    class Builder;

    KnudsenCodec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length);

    ~KnudsenCodec() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
    T _characteristic_length = T(1);

    DeviceBuffer<T> d_knudsen_values {};
};

template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    DomainHostPtr<T> _domain {};

    T _characteristic_length = T(1);
};

}

namespace atlas {

template <typename T>
using KnudsenCodec = system::KnudsenCodec<T>;

template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<system::KnudsenCodec<T>>;

template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<system::KnudsenCodec<T>>;

}

#include <atlas/codec/knudsen_codec.hpp>