#pragma once

#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

enum class CodecType : int {
    single,
    knudsen,
    deep_learning,
};

template <typename T>
struct CodecDeviceProbe {

    CodecType type;

    int* allocated_system;
};

template <typename T>
class Codec {
public:
    Codec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Codec(DomainHostPtr<T> domain);

    virtual ~Codec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe)
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>
    make_device_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual CodecType
    type() const noexcept = 0;

private:
    std::uint64_t _probe_count = 0;

    DomainHostPtr<T> _domain {};

    DeviceBuffer<int> d_allocated_system;
};

}

namespace atlas {

template <typename T>
using Codec = system::Codec<T>;

template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

}

#include <atlas/codec/codec.hpp>
