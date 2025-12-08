#pragma once
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {
    template <typename T>
    class Remover {
    public:
        Remover() = default;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Remover(T x_min, T x_max,
                T y_min, T y_max,
                T z_min, T z_max);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        operator()(const ParticleDeviceProbe<T>& data, int& active) const;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_x_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_x_max(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_y_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_y_max(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_z_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_z_max(T v);
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        x_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        x_max() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        y_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        y_max() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        z_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        z_max() const;

    private:
        T x_min_ { T(0) };
        T x_max_ { T(0) };
        T y_min_ { T(0) };
        T y_max_ { T(0) };
        T z_min_ { T(0) };
        T z_max_ { T(0) };
    };
}

template <typename T>
using Remover = system::Remover<T>;
template <typename T>
using RemoverHostPtr = host_shared_ptr<Remover<T>>;
}

#include <atlas/remover/remover.hpp>