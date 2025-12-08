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
        ParticleSystem();
        explicit ParticleSystem(int capacity);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_emitter(const EmitterHostPtr<T>& emitter_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_remover(const RemoverHostPtr<T>& remover_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_advector(const AdvectorHostPtr<T>& advector_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_solver(const SolverHostPtr<T>& solver_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_emitter(const Emitter<T>& emitter_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_remover(const Remover<T>& remover_);
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_advector(const Advector<T>& advector_);
        ATLAS_HOST ATLAS_FORCE_INLINE const ParticleDataHostPtr<T>&
        particles() const;
        ATLAS_HOST ATLAS_FORCE_INLINE const EmitterHostPtr<T>&
        emitter() const;
        ATLAS_HOST ATLAS_FORCE_INLINE const RemoverHostPtr<T>&
        remover() const;
        ATLAS_HOST ATLAS_FORCE_INLINE const AdvectorHostPtr<T>&
        advector() const;
        ATLAS_HOST ATLAS_FORCE_INLINE const SolverHostPtr<T>&
        solver() const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        update();
        ATLAS_HOST ATLAS_FORCE_INLINE void
        emit(const ParticleDeviceProbe<T>& data, int& active) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        advect(const ParticleDeviceProbe<T>& data, T dt, int& active) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        solve(const ParticleDeviceProbe<T>& data, T dt, int& active) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        remove(const ParticleDeviceProbe<T>& data, int& active) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
        capacity() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
        active() const;

    private:
        T _dt { T(0.000001f) };
        int _capacity { 0 };
        int _active { 0 };
        ParticleDataHostPtr<T> _particle_data = nullptr;
        ParticleDeviceProbe<T> _device_probe {};
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