#pragma once
#include <atlas/core/macros.h>
#include <atlas/data/particle_data.h>
#include <atlas/domain/domain.h>

namespace atlas::system {
template <typename T>
class Measure {
public:

    Measure()  = default;
    virtual ~Measure() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure(DomainDeviceProbe<T> domain, ParticleDeviceProbe<T> particle)
        = 0;
};
}