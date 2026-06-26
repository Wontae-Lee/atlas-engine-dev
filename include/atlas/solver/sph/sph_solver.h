#pragma once

#include <atlas/core/macros.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>

namespace atlas {

template <typename T>
class SphSolver final : public Solver<T> {
public:
    using SphSolverProbe = atlas::SphProbe<T>;

    class Builder;

    SphSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SphSolver(UniverseHostPtr<T> universe,
              FluidHostPtr<T> fluid,
              SearcherHostPtr<T> searcher,
              SphKernelType kernel_type = SphKernelType::standard) noexcept;

    ~SphSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_fields();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_fields();

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density(const MaterialProperties<T>& property) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient(const MaterialProperties<T>& property) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_density();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    count_particles();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    accelerate(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

private:
    SphSolverProbe _probe {};

    SphKernel<T> _kernel {};

    DeviceBuffer<T> _density {};

    DeviceBuffer<T> _pressure {};

    DeviceBuffer<Vector3<T>> _acceleration {};
};

template <typename T>
class SphSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SphSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    SphKernelType _kernel_type { SphKernelType::standard };
};

}

namespace atlas {

template <typename T>
using SphSolverHostPtr = atlas::host_shared_ptr<atlas::SphSolver<T>>;

template <typename T>
using SphSolverDevicePtr = atlas::device_shared_ptr<atlas::SphSolver<T>>;

}

#include <atlas/solver/sph/sph_solver.hpp>