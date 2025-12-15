#pragma once
#include <atlas/advector/advector.h>
#include <atlas/emitter/emitter.h>
#include <atlas/memory/memory.h>
#include <atlas/remover/remover.h>
#include <atlas/solver/solver.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {
    template <typename T>
    class ParticleSystem {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE
        ParticleSystem();

        explicit
        ParticleSystem(
            T dt,
            const EmitterHostPtr<T>& emitter   = nullptr,
            const RemoverHostPtr<T>& remover   = nullptr,
            const AdvectorHostPtr<T>& advector = nullptr,
            const SolverHostPtr<T>& solver     = nullptr,
            int capacity                       = 200000);

        ATLAS_HOST ATLAS_FORCE_INLINE const ParticleDataHostPtr<T>&
        particles() const;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        update();

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
        alive() const;

    private:
        T _dt{ 1.0f };
        int _capacity{ 0 };
        ParticleDataHostPtr<T> _particle_data = nullptr;
        ParticleDeviceProbe<T> _device_probe{};
        SolverHostPtr<T> _solver     = nullptr;
        EmitterHostPtr<T> _emitter   = nullptr;
        RemoverHostPtr<T> _remover   = nullptr;
        AdvectorHostPtr<T> _advector = nullptr;
    };
}

template <typename T>
using ParticleSystem = system::ParticleSystem<T>;
template <typename T>
using ParticleSystemHostPtr = host_shared_ptr<ParticleSystem<T>>;
}

#include <atlas/system/particle_system.hpp>