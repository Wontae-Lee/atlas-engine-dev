#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcEnergyExchangeSolver final : public DsmcSolver<T> {
public:
    using Base = DsmcSolver<T>;
    using Probe = typename Base::Probe;
    using Base::Base;

    class Builder;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         int local_collision,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sample_unit(int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sample_bl(T exp_1, T exp_2, int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rotational_relaxation_probability(const MaterialProperties<T>& material,
                                      T collision_energy,
                                      T omega) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    vibrational_relaxation_probability(const MaterialProperties<T>& material,
                                       T collision_energy,
                                       T omega) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static T
    exchange_internal_energy(const Probe& probe,
                             int cell,
                             int local_collision,
                             int particle_i,
                             int particle_j,
                             std::size_t species_i,
                             std::size_t species_j,
                             T relative_speed_squared) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    exchange_particle_internal_energy(const Probe& probe,
                                      int cell,
                                      int local_collision,
                                      int particle,
                                      const MaterialProperties<T>& material,
                                      T omega,
                                      std::uint64_t salt_base,
                                      T& e_dispose) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    rescale_relative_velocity(Vector3<T>& lhs_velocity,
                              Vector3<T>& rhs_velocity,
                              const MaterialProperties<T>& lhs,
                              const MaterialProperties<T>& rhs,
                              T translational_energy) noexcept;

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_flattened_collision(const DeviceBuffer<int>* allocated_solver, int index) override;
};

template <typename T>
class DsmcEnergyExchangeSolver<T>::Builder final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcEnergyExchangeSolver<T>
    build() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcEnergyExchangeSolver<T>>
    make_host_shared() const;

private:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SearcherHostPtr<T> _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcEnergyExchangeSolver = atlas::system::DsmcEnergyExchangeSolver<T>;

template <typename T>
using DsmcEnergyExchangeSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcEnergyExchangeSolver<T>>;

template <typename T>
using DsmcEnergyExchangeSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcEnergyExchangeSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_energy_exchange_solver.hpp>
