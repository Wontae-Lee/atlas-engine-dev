#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/sparta/sparta_vss_kernel.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

template <typename T>
class DsmcSpartaSolver final : public Solver<T> {
public:
    DsmcSpartaSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSpartaSolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ~DsmcSpartaSolver() override = default;

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
    DsmcProbe<T> _probe {};
    SpartaVssKernel<T> _kernel {};
    std::uint64_t _collision_seed = 0;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcSpartaSolver = atlas::system::DsmcSpartaSolver<T>;

template <typename T>
using DsmcSpartaSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSpartaSolver<T>>;

template <typename T>
using DsmcSpartaSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSpartaSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/sparta/dsmc_sparta_solver.hpp>
