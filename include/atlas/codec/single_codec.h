#pragma once

#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class SingleCodec final : public Codec<T> {
public:
    class Builder;

    SingleCodec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit SingleCodec(const DomainHostPtr<T>& domain);

    ~SingleCodec() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const Universe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const Universe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;
};

template <typename T>
class SingleCodec<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SingleCodec<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SingleCodec<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    DomainHostPtr<T> _domain {};
};

}

namespace atlas {

template <typename T>
using SingleCodec = system::SingleCodec<T>;

template <typename T>
using SingleCodecHostPtr = atlas::host_shared_ptr<system::SingleCodec<T>>;

template <typename T>
using SingleCodecDevicePtr = atlas::device_shared_ptr<system::SingleCodec<T>>;

}

#include <atlas/codec/single_codec.hpp>