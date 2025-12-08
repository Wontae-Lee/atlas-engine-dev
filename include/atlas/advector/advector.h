#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/collider/collider.h>
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {
    template <typename T>
    class Advector {
    public:
        Advector() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_collider(const ColliderHostPtr<T>& collider);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_collider(const Collider<T>& collider);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        operator()(const ParticleDeviceProbe<T>& probe, T dt, int& active) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        time_integration(const ParticleDeviceProbe<T>& probe, T dt, int& active) const;

    private:
        ColliderHostPtr<T> _collider;
    };
}

template <typename T>
using Advector = system::Advector<T>;
template <typename T>
using AdvectorHostPtr = host_shared_ptr<Advector<T>>;
template <typename T>
using AdvectorDevicePtr = device_shared_ptr<Advector<T>>;
}

#include <atlas/advector/advector.hpp>
