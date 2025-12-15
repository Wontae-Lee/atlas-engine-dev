#pragma once
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace solver {
    template <typename T>
    class Solver {
    public:
        Solver()          = default;
        virtual ~Solver() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        operator()(const system::ParticleDeviceProbe<T>& data, T dt)
            = 0;

    protected:
        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        solve(const system::ParticleDeviceProbe<T>& data, T dt)
            = 0;
    };
}

template <typename T>
using Solver = solver::Solver<T>;
template <typename T>
using SolverHostPtr = atlas::host_shared_ptr<solver::Solver<T>>;
template <typename T>
using SolverDevicePtr = atlas::device_shared_ptr<solver::Solver<T>>;
}