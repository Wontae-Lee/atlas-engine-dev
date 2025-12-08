#pragma once

#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/dsmc/dsmc_data.h>
#include <atlas/solver/dsmc/metrics/g_ref_estimator.h>
#include <atlas/solver/solver.h>
namespace atlas {
namespace solver {
    enum class DsmcWorkDistribution {
        PerCell,
        PerCollision
    };
    template <typename T>
    class DsmcSolver final : public Solver<T> {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE
        DsmcSolver();
        ~DsmcSolver() override = default;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        operator()(const system::ParticleDeviceProbe<T>& data,
                   T dt,
                   int& active) override;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        solve(const system::ParticleDeviceProbe<T>& data,
              T dt,
              int& active) override;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_work_distribution(const DsmcWorkDistribution dist) {
            _dsmc_work_distribution = dist;
        }
        ATLAS_HOST ATLAS_FORCE_INLINE void
        compute_g_ref(const system::SpatialHashProbe<T>& neighbor_probe,
                      const DsmcDeviceProbe<T>& dsmc_probe,
                      const system::ParticleDeviceProbe<T>& data,
                      int n_cells) {
            if (_g_ref_estimator) {
                _g_ref_estimator->compute(neighbor_probe, dsmc_probe, data, n_cells);
            }
        }
        ATLAS_HOST ATLAS_FORCE_INLINE void
        count_collisions(const system::SpatialHashProbe<T>& neighbor_probe,
                         const DsmcDeviceProbe<T>& dsmc_probe,
                         T dt);
        void
        collide_per_cell(const system::SpatialHashProbe<T>& neighbor_probe,
                         const DsmcDeviceProbe<T>& dsmc_probe,
                         const system::ParticleDeviceProbe<T>& data,
                         T dt);
        void
        collide_per_collision(const system::SpatialHashProbe<T>& neighbor_probe,
                              const DsmcDeviceProbe<T>& dsmc_probe,
                              const system::ParticleDeviceProbe<T>& data,
                              T dt);

    private:
        SpatialHashingSearcherHostPtr<T> _searcher;
        DsmcDataHostPtr<T> _dsmc_data;
        dsmc::GRefEstimatorHostPtr<T> _g_ref_estimator;
        DsmcWorkDistribution _dsmc_work_distribution = DsmcWorkDistribution::PerCollision;
    };
}
template <typename T>
using DsmcSolver = solver::DsmcSolver<T>;
template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<solver::DsmcSolver<T>>;
template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<solver::DsmcSolver<T>>;
}
#include <atlas/solver/dsmc/dsmc_solver.hpp>