#pragma once

#include <atlas/core/macros.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>

namespace atlas {

class SphSolver final : public Solver {
public:
    using SphSolverProbe = atlas::SphProbe;

    class Builder;

    SphSolver() = default;

    ATLAS_HOST
    SphSolver(UniverseHostPtr universe,
              FluidHostPtr fluid,
              SearcherHostPtr searcher,
              SphKernelType kernel_type = SphKernelType::standard) noexcept;

    ~SphSolver() override = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_HOST SphKernelType
    kernel_type() const noexcept;

    ATLAS_HOST void
    solve(float dt) override;

    ATLAS_HOST void
    solve(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    ATLAS_HOST void
    ensure_states();

    ATLAS_HOST bool
    initialize_context() noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

    ATLAS_HOST bool
    prepare_fields();

    ATLAS_HOST void
    reset_fields();

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    rest_density(const MaterialProperties& property) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    pressure_coefficient(const MaterialProperties& property) noexcept;

    ATLAS_HOST void
    estimate_density();

    ATLAS_HOST void
    count_particles();

    ATLAS_HOST void
    accelerate(float dt);

    ATLAS_HOST void
    update();

private:
    SphSolverProbe _probe {};

    SphKernel _kernel {};

    DeviceBuffer<float> _density {};

    DeviceBuffer<float> _pressure {};

    DeviceBuffer<Float3> _acceleration {};
};

class SphSolver::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    ATLAS_NODISCARD ATLAS_HOST SphSolver
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SphSolver>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    SphKernelType _kernel_type { SphKernelType::standard };
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphSolver::rest_density(const MaterialProperties& property) noexcept {
    if (property.rest_density.has_value() && *property.rest_density > 0.0f) {
        return *property.rest_density;
    }

    return 1.0f;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphSolver::pressure_coefficient(const MaterialProperties& property) noexcept {
    return property.pressure_coefficient.value_or(0.0f);
}

using SphSolverHostPtr = atlas::host_shared_ptr<atlas::SphSolver>;

using SphSolverDevicePtr = atlas::device_shared_ptr<atlas::SphSolver>;

}
