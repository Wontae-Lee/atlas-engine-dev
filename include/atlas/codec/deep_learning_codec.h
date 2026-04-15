#pragma once

#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class DeepLearningCodec final : public Codec<T> {
public:
    class Builder;

    DeepLearningCodec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit DeepLearningCodec(const DomainHostPtr<T>& domain);

    ~DeepLearningCodec() override = default;

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

private:
};

template <typename T>
class DeepLearningCodec<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeepLearningCodec<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DeepLearningCodec<T>>
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
using DeepLearningCodec = system::DeepLearningCodec<T>;

template <typename T>
using DeepLearningCodecHostPtr = atlas::host_shared_ptr<system::DeepLearningCodec<T>>;

template <typename T>
using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<system::DeepLearningCodec<T>>;

}

#include <atlas/codec/deep_learning_codec.hpp>