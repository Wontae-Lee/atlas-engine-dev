#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/piclas/piclas_vhs_kernel.h>
#include <atlas/solver/solver.h>

#include <cstddef>
#include <cstdint>

namespace atlas::system {

template <typename T>
class DsmcPiclasSolver final : public Solver<T> {
public:
    DsmcPiclasSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcPiclasSolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ~DsmcPiclasSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_states();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt);

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    species_count(const DsmcProbe<T>& probe, int begin, int end, std::size_t species) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    pair_case_count(const DsmcProbe<T>& probe,
                    int cell,
                    int begin,
                    int count,
                    std::size_t species_i,
                    std::size_t species_j,
                    std::uint64_t seed) noexcept;

private:
    DsmcProbe<T> _probe {};
    PiclasVhsKernel<T> _kernel {};
    std::uint64_t _collision_seed = 0;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcPiclasSolver = atlas::system::DsmcPiclasSolver<T>;

template <typename T>
using DsmcPiclasSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcPiclasSolver<T>>;

template <typename T>
using DsmcPiclasSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcPiclasSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/piclas/dsmc_piclas_solver.hpp>
