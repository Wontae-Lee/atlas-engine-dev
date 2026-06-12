#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/sph/sph_solver.h>

namespace atlas {

template <typename T>
class SphGatewaySolver : public Solver<T> {
public:
    using SphSolverProbe = typename SphSolver<T>::SphSolverProbe;

    class Builder;

    SphGatewaySolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SphGatewaySolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SearcherHostPtr<T> searcher,
                     SphKernelType kernel_type = SphKernelType::standard,
                     int group_particle_count  = 5) noexcept;

    ~SphGatewaySolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    group_particle_count() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_group_fields();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_group_representatives(const DeviceBuffer<int>* allocated_solver, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, int index);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density_for(const MaterialProperties<T>& property) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MaterialProperties<T>& property) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    group_count_for_cell(int particle_count, int group_particle_count) noexcept;

protected:
    SphSolverProbe _probe {};

    SphKernel<T> _kernel {};

    int _group_particle_count { 5 };

    DeviceBuffer<int> _cell_group_count {};

    DeviceBuffer<Vector3<T>> _group_position {};

    DeviceBuffer<Vector3<T>> _group_velocity {};

    DeviceBuffer<Vector3<T>> _group_updated_position {};

    DeviceBuffer<Vector3<T>> _group_updated_velocity {};

    DeviceBuffer<T> _group_mass {};

    DeviceBuffer<T> _group_density {};

    DeviceBuffer<T> _group_pressure {};

    DeviceBuffer<int> _group_member_count {};

    DeviceBuffer<std::size_t> _group_species {};
};

template <typename T>
class SphGatewaySolver<T>::Builder final {
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

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_group_particle_count(int group_particle_count) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SphGatewaySolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphGatewaySolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    SphKernelType _kernel_type { SphKernelType::standard };

    int _group_particle_count { 5 };
};

}

namespace atlas {

template <typename T>
using SphGatewaySolverHostPtr = atlas::host_shared_ptr<atlas::SphGatewaySolver<T>>;

template <typename T>
using SphGatewaySolverDevicePtr = atlas::device_shared_ptr<atlas::SphGatewaySolver<T>>;

}

#include <atlas/solver/sph/sph_gateway_solver.hpp>