#ifndef ATLAS_ENGINE_DEV_DSMC_H
#define ATLAS_ENGINE_DEV_DSMC_H

#include <atlas/solver/solver.h>
namespace atlas {
namespace solver {

    template <typename T>
    class DSMCSolver final : public Solver<T> {
    public:
        DSMCSolver()           = default;
        ~DSMCSolver() override = default;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        solve(const system::ParticleDeviceProbe<T>& data, T dt, int& active) override;
    };

}

template <typename T>
using DSMCSolver = solver::DSMCSolver<T>;

template <typename T>
using DSMCSolverHostPtr = atlas::host_shared_ptr<solver::DSMCSolver<T>>;

template <typename T>
using DSMCSolverDevicePtr = atlas::device_shared_ptr<solver::DSMCSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc.hpp>

#endif