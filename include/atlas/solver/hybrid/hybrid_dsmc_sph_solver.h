#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/hybrid/hybrid_dsmc_sph_probe.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/sph/sph_solver.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace atlas {

template <typename T>
class HybridDsmcSphSolver final : public Solver<T> {
public:
    using SphSolverProbe = typename SphSolver<T>::SphSolverProbe;
    using DsmcProbe      = ::atlas::DsmcProbe<T>;
    using HybridProbe    = atlas::HybridDsmcSphProbe<T>;

    class Builder;

public:
    HybridDsmcSphSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    HybridDsmcSphSolver(UniverseHostPtr<T> universe,
                         FluidHostPtr<T> fluid,
                         SearcherHostPtr<T> searcher,
                         T grouping_length,
                         int sph_particle_threshold,
                         SphKernelType sph_kernel_type = SphKernelType::standard,
                         DsmcKernelType dsmc_kernel_type = DsmcKernelType::hard_sphere,
                         bool pairing_without_replacement = false) noexcept;

    ~HybridDsmcSphSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    grouping_length() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    sph_particle_threshold() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    sph_kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    dsmc_kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    pairing_without_replacement() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_fields();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_states();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify_particles();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_sph_groups();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_dsmc_groups();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_grouped_dsmc(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure_grouped_dsmc_statistics(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_random_grouped_dsmc_collisions();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_grouped_dsmc_collisions_without_replacement();

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density_for(const MaterialProperties<T>& property) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MaterialProperties<T>& property) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    select_pair_offsets_without_replacement(int& lhs_local,
                                            int& rhs_local,
                                            int local_pair,
                                            int count,
                                            int selector,
                                            std::uint64_t seed) noexcept;

private:
    HybridProbe _probe {};
    SphKernel<T> _sph_kernel {};
    DsmcKernel<T> _dsmc_kernel {};
    T _grouping_length {};
    int _sph_particle_threshold { 5 };
    std::uint64_t _collision_seed {};
    bool _pairing_without_replacement {};

    DeviceBuffer<int> _sph_candidate {};
    DeviceBuffer<int> _group_owner {};
    DeviceBuffer<int> _group_member_count {};
    DeviceBuffer<int> _dsmc_particle_count {};
    DeviceBuffer<int> _dsmc_group_owner {};
    DeviceBuffer<int> _dsmc_group_member_count {};
    DeviceBuffer<int> _dsmc_collision_count {};
    DeviceBuffer<T> _dsmc_max_relative_speed {};
    DeviceBuffer<T> _dsmc_max_sigma_g {};

    DeviceBuffer<Vector3<T>> _group_position {};
    DeviceBuffer<Vector3<T>> _group_velocity {};
    DeviceBuffer<Vector3<T>> _group_updated_velocity {};
    DeviceBuffer<T> _group_mass {};
    DeviceBuffer<T> _group_density {};
    DeviceBuffer<T> _group_pressure {};
    DeviceBuffer<std::size_t> _group_species {};
};

template <typename T>
class HybridDsmcSphSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_grouping_length(T grouping_length) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sph_particle_threshold(int threshold) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sph_kernel_type(SphKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dsmc_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_pairing_without_replacement(bool enabled) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE HybridDsmcSphSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<HybridDsmcSphSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SearcherHostPtr<T> _searcher {};
    T _grouping_length {};
    int _sph_particle_threshold { 5 };
    SphKernel<T> _sph_kernel {};
    DsmcKernel<T> _dsmc_kernel {};
    bool _pairing_without_replacement {};
};

} // namespace atlas

namespace atlas {
template <typename T>
using HybridDsmcSphSolverHostPtr = atlas::host_shared_ptr<atlas::HybridDsmcSphSolver<T>>;

template <typename T>
using HybridDsmcSphSolverDevicePtr = atlas::device_shared_ptr<atlas::HybridDsmcSphSolver<T>>;

} // namespace atlas

#include <atlas/solver/hybrid/hybrid_dsmc_sph_solver.hpp>
