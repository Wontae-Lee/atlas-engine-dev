#ifndef ATLAS_ENGINE_DEV_SOLVER_H
#define ATLAS_ENGINE_DEV_SOLVER_H

#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace solver {

    template <typename T>
    class Solver {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE
        Solver();
        virtual ~Solver() = default;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        operator()(const system::ParticleDeviceProbe<T>& data, T dt, int& active);

    protected:
        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        solve(const system::ParticleDeviceProbe<T>& data, T dt, int& active)
            = 0;

    private:
        SearcherHostPtr<T> _searcher;
    };
}

template <typename T>
using Solver = solver::Solver<T>;

template <typename T>
using SolverHostPtr = atlas::host_shared_ptr<solver::Solver<T>>;

template <typename T>
using SolverDevicePtr = atlas::device_shared_ptr<solver::Solver<T>>;

}

#include <atlas/solver/solver.hpp>

#endif