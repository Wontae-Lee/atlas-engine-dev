#pragma once
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {
    template <typename T>
    class Emitter {
    public:
        Emitter() = default;

        void
        set_emit_per_step(int n) { emit_per_step = n; }

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Emitter(T x_spawn_min, T x_spawn_max,
                T y_spawn_min, T y_spawn_max,
                T z_spawn_min, T z_spawn_max,
                T inject_vx_mean, T inject_vx_jit,
                T inject_vt_jit);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        operator()(ParticleDeviceProbe<T>& data);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_x_spawn_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_x_spawn_max(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_y_spawn_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_y_spawn_max(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_z_spawn_min(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_z_spawn_max(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_inject_vx_mean(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_inject_vx_jit(T v);
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_inject_vt_jit(T v);
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        x_spawn_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        x_spawn_max() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        y_spawn_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        y_spawn_max() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        z_spawn_min() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        z_spawn_max() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        inject_vx_mean() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        inject_vx_jit() const;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        inject_vt_jit() const;

    private:
        int emit_per_step{ 1000 };
        int total_emitted{ 0 };
        T x_spawn_min_{ T(0) };
        T x_spawn_max_{ T(0) };
        T y_spawn_min_{ T(0) };
        T y_spawn_max_{ T(0) };
        T z_spawn_min_{ T(0) };
        T z_spawn_max_{ T(0) };
        T inject_vx_mean_{ T(0) };
        T inject_vx_jit_{ T(0) };
        T inject_vt_jit_{ T(0) };
    };
}

template <typename T>
using Emitter = system::Emitter<T>;
template <typename T>
using EmitterHostPtr = host_shared_ptr<Emitter<T>>;
}

#include <atlas/emitter/emitter.hpp>