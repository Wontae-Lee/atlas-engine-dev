// <atlas/system/source.h>
#pragma once

#include <atlas/data/particle_data.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

template <typename T>
class Source {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    Source()
        = default;
    ATLAS_HOST ATLAS_FORCE_INLINE ~Source() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Source(const ParticleDeviceProbe<T>& probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_probe(const ParticleDeviceProbe<T>& probe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>
    device_probe() const noexcept;

private:
    ParticleDeviceProbe<T> _probe {};
};

} // namespace atlas::system

namespace atlas {
template <typename T>
using Source = system::Source<T>;

template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<system::Source<T>>;
} // namespace atlas

#include <atlas/source/source.hpp>
