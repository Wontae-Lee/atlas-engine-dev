#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

#include <cmath>

namespace atlas {

class DsmcEnergyExchangeSolver final : public DsmcSolver {
public:
    using Base  = DsmcSolver;
    using Probe = Base::Probe;
    using Base::Base;

    class Builder;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 float max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         int local_collision,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         float max_sigma_g) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    sample_unit(int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    sample_bl(float exp_1, float exp_2, int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    rotational_relaxation_probability(const MaterialProperties& material,
                                      float collision_energy,
                                      float omega) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    vibrational_relaxation_probability(const MaterialProperties& material,
                                       float collision_energy,
                                       float omega) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    exchange_internal_energy(const Probe& probe,
                             int cell,
                             int local_collision,
                             int particle_i,
                             int particle_j,
                             std::size_t species_i,
                             std::size_t species_j,
                             float relative_speed_squared) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    exchange_particle_internal_energy(const Probe& probe,
                                      int cell,
                                      int local_collision,
                                      int particle,
                                      const MaterialProperties& material,
                                      float omega,
                                      std::uint64_t salt_base,
                                      float& e_dispose) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    rescale_relative_velocity(Float3& lhs_velocity,
                              Float3& rhs_velocity,
                              const MaterialProperties& lhs,
                              const MaterialProperties& rhs,
                              float translational_energy) noexcept;

protected:
    ATLAS_HOST void
    apply_flattened_collision(const DeviceBuffer<int>* allocated_solver, int index) override;

public:
    ATLAS_HOST void
    apply_cell_energy_collisions(const int* allocated_solver_ptr, int index);

    ATLAS_HOST void
    launch_flattened_energy_collisions();
};

class DsmcEnergyExchangeSolver::Builder final {
public:
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST Builder&
    with_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST void
    validate() const;

    ATLAS_NODISCARD ATLAS_HOST DsmcEnergyExchangeSolver
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<DsmcEnergyExchangeSolver>
    make_host_shared() const;

private:
    UniverseHostPtr _universe {};
    FluidHostPtr _fluid {};
    SearcherHostPtr _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcEnergyExchangeSolver::collide_pair(const Probe& probe,
                                       const int cell,
                                       const int local_collision,
                                       const int begin,
                                       const int end,
                                       const int lhs_local,
                                       const int rhs_local,
                                       const float max_sigma_g) noexcept {
    const int particle_i = DsmcSolver::particle_at(lhs_local, begin, end, probe.particle_count, probe.indices_ptr);
    const int particle_j = DsmcSolver::particle_at(rhs_local, begin, end, probe.particle_count, probe.indices_ptr);

    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return DsmcEnergyExchangeSolver::collide_indexed_pair(
        probe,
        cell,
        local_collision,
        stream,
        particle_i,
        particle_j,
        max_sigma_g);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcEnergyExchangeSolver::collide_indexed_pair(const Probe& probe,
                                               const int cell,
                                               const int local_collision,
                                               const std::uint64_t stream,
                                               const int particle_i,
                                               const int particle_j,
                                               const float max_sigma_g) noexcept {
    if (particle_i < 0 || particle_j < 0 || particle_i == particle_j) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];
    if (species_i >= static_cast<std::size_t>(probe.species_count)
        || species_j >= static_cast<std::size_t>(probe.species_count)) {
        return false;
    }

    Float3 lhs_velocity                = probe.velocity_ptr[particle_i];
    Float3 rhs_velocity                = probe.velocity_ptr[particle_j];
    const float relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const float sigma_g                = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);
    if (!(sigma_g > 0.0f)) {
        return false;
    }

    float local_max_sigma_g = max_sigma_g;
    if (sigma_g > local_max_sigma_g) {
        local_max_sigma_g           = sigma_g;
        probe.max_sigma_g_ptr[cell] = sigma_g;
    }

    float accept_probability = sigma_g / local_max_sigma_g;
    if (accept_probability > 1.0f) {
        accept_probability = 1.0f;
    }

    const float accept_sample = atlas::sample_hashed_unit_interval(
        cell,
        probe.collision_seed + stream + atlas::DSMC_COLLISION_ACCEPT_SALT);
    if (accept_sample >= accept_probability) {
        return false;
    }

    const float post_translational_energy = DsmcEnergyExchangeSolver::exchange_internal_energy(
        probe,
        cell,
        local_collision,
        particle_i,
        particle_j,
        species_i,
        species_j,
        relative_speed_squared);

    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    DsmcEnergyExchangeSolver::rescale_relative_velocity(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j],
        post_translational_energy);

    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;
    return true;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcEnergyExchangeSolver::sample_unit(const int cell,
                                      const int local_collision,
                                      const std::uint64_t seed,
                                      const std::uint64_t salt) noexcept {
    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return atlas::sample_hashed_unit_interval(cell, seed + stream + salt);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcEnergyExchangeSolver::sample_bl(const float exp_1,
                                    const float exp_2,
                                    const int cell,
                                    const int local_collision,
                                    const std::uint64_t seed,
                                    const std::uint64_t salt) noexcept {
    if (!(exp_1 > 0.0f) || !(exp_2 > 0.0f)) {
        return 0.0f;
    }

    const float exp_sum = exp_1 + exp_2;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const float x = DsmcEnergyExchangeSolver::sample_unit(
            cell,
            local_collision,
            seed,
            salt + static_cast<std::uint64_t>(attempt) * 2u);
        const float y = std::pow(x * exp_sum / exp_1, exp_1)
            * std::pow((1.0f - x) * exp_sum / exp_2, exp_2);
        const float accept = DsmcEnergyExchangeSolver::sample_unit(
            cell,
            local_collision,
            seed,
            salt + static_cast<std::uint64_t>(attempt) * 2u + 1u);
        if (accept <= y) {
            return x;
        }
    }
    return 0.5f;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcEnergyExchangeSolver::rotational_relaxation_probability(const MaterialProperties& material,
                                                            const float collision_energy,
                                                            const float omega) noexcept {
    const int dof = material.rotational_dof.value_or(0);
    if (dof <= 0) {
        return 0.0f;
    }

    if (material.rotational_relaxation_c1.has_value()
        && material.rotational_relaxation_c2.has_value()
        && material.rotational_relaxation_c3.has_value()
        && collision_energy > 0.0f) {
        const float denominator = atlas::boltzmann_constant
            * (2.5f - omega + static_cast<float>(dof) * 0.5f);
        if (denominator > 0.0f) {
            const float tr = collision_energy / denominator;
            if (tr > 0.0f) {
                const float probability = (1.0f
                                           + material.rotational_relaxation_c2.value() / atlas::sqrt_nonnegative(tr)
                                           + material.rotational_relaxation_c3.value() / tr)
                    / material.rotational_relaxation_c1.value();
                return probability < 0.0f ? 0.0f : (probability > 1.0f ? 1.0f : probability);
            }
        }
    }

    return material.rotational_relaxation_probability.value_or(0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcEnergyExchangeSolver::vibrational_relaxation_probability(const MaterialProperties& material,
                                                             const float collision_energy,
                                                             const float omega) noexcept {
    if (material.vibrational_dof.value_or(0) <= 0) {
        return 0.0f;
    }

    if (material.vibrational_relaxation_c1.has_value()
        && material.vibrational_relaxation_c2.has_value()
        && collision_energy > 0.0f) {
        const float denominator = atlas::boltzmann_constant * (3.5f - omega);
        if (denominator > 0.0f) {
            const float tr = collision_energy / denominator;
            if (tr > 0.0f) {
                const float probability = 1.0f
                    / (material.vibrational_relaxation_c1.value() / std::pow(tr, omega)
                       * std::exp(material.vibrational_relaxation_c2.value() / std::pow(tr, 1.0f / 3.0f)));
                return probability < 0.0f ? 0.0f : (probability > 1.0f ? 1.0f : probability);
            }
        }
    }

    return material.vibrational_relaxation_probability.value_or(0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcEnergyExchangeSolver::exchange_internal_energy(const Probe& probe,
                                                   const int cell,
                                                   const int local_collision,
                                                   const int particle_i,
                                                   const int particle_j,
                                                   const std::size_t species_i,
                                                   const std::size_t species_j,
                                                   const float relative_speed_squared) noexcept {
    const auto& lhs_material = probe.properties_ptr[species_i];
    const auto& rhs_material = probe.properties_ptr[species_j];
    const auto pair          = DsmcKernel::pair_parameters(lhs_material, rhs_material);
    if (!pair.valid) {
        return 0.0f;
    }

    float e_dispose = 0.5f * pair.reduced_mass * relative_speed_squared;
    if (probe.internal_energy_ptr == nullptr) {
        return e_dispose;
    }

    DsmcEnergyExchangeSolver::exchange_particle_internal_energy(
        probe,
        cell,
        local_collision,
        particle_i,
        lhs_material,
        pair.viscosity_index,
        0x7f4a7c15ull,
        e_dispose);
    DsmcEnergyExchangeSolver::exchange_particle_internal_energy(
        probe,
        cell,
        local_collision,
        particle_j,
        rhs_material,
        pair.viscosity_index,
        0x94d049bbull,
        e_dispose);

    return e_dispose > 0.0f ? e_dispose : 0.0f;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcEnergyExchangeSolver::exchange_particle_internal_energy(const Probe& probe,
                                                            const int cell,
                                                            const int local_collision,
                                                            const int particle,
                                                            const MaterialProperties& material,
                                                            const float omega,
                                                            const std::uint64_t salt_base,
                                                            float& e_dispose) noexcept {
    auto energy = probe.internal_energy_ptr[particle];

    const int rot_dof           = material.rotational_dof.value_or(0);
    const float rot_probability = DsmcEnergyExchangeSolver::rotational_relaxation_probability(
        material,
        e_dispose + energy.rotational,
        omega);
    if (rot_dof > 0
        && DsmcEnergyExchangeSolver::sample_unit(cell, local_collision, probe.collision_seed, salt_base) <= rot_probability) {
        e_dispose += energy.rotational;
        if (rot_dof == 2) {
            const float exponent = 2.5f - omega;
            const float u        = DsmcEnergyExchangeSolver::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 1u);
            energy.rotational    = exponent > 0.0f
                   ? (1.0f - std::pow(u, 1.0f / exponent)) * e_dispose
                   : 0.0f;
        } else {
            energy.rotational = e_dispose * DsmcEnergyExchangeSolver::sample_bl(static_cast<float>(rot_dof) * 0.5f - 1.0f, 1.5f - omega, cell, local_collision, probe.collision_seed, salt_base + 2u);
        }
        e_dispose -= energy.rotational;
    } else if (rot_dof <= 0) {
        energy.rotational = 0.0f;
    }

    const int vib_dof           = material.vibrational_dof.value_or(0);
    const float vib_probability = DsmcEnergyExchangeSolver::vibrational_relaxation_probability(
        material,
        e_dispose + energy.vibrational,
        omega);
    if (vib_dof > 0
        && DsmcEnergyExchangeSolver::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 17u) <= vib_probability) {
        e_dispose += energy.vibrational;
        if (vib_dof == 2) {
            const float exponent = 2.5f - omega;
            const float u        = DsmcEnergyExchangeSolver::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 18u);
            energy.vibrational   = exponent > 0.0f
                  ? (1.0f - std::pow(u, 1.0f / exponent)) * e_dispose
                  : 0.0f;
        } else {
            energy.vibrational = e_dispose * DsmcEnergyExchangeSolver::sample_bl(static_cast<float>(vib_dof) * 0.5f - 1.0f, 1.5f - omega, cell, local_collision, probe.collision_seed, salt_base + 19u);
        }
        e_dispose -= energy.vibrational;
    } else if (vib_dof <= 0) {
        energy.vibrational = 0.0f;
    }

    energy.translational                = e_dispose;
    probe.internal_energy_ptr[particle] = energy;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcEnergyExchangeSolver::rescale_relative_velocity(Float3& lhs_velocity,
                                                    Float3& rhs_velocity,
                                                    const MaterialProperties& lhs,
                                                    const MaterialProperties& rhs,
                                                    const float translational_energy) noexcept {
    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
        || !(translational_energy > 0.0f)) {
        return;
    }

    const float reduced_mass = lhs_mass * rhs_mass / mass_sum;
    if (!(reduced_mass > 0.0f)) {
        return;
    }

    const Float3 relative = lhs_velocity - rhs_velocity;
    const float speed     = relative.length();
    if (!(speed > 0.0f)) {
        return;
    }

    const float target_speed        = atlas::sqrt_nonnegative(2.0f * translational_energy / reduced_mass);
    const Float3 scattered_relative = relative * (target_speed / speed);
    const Float3 center             = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

using DsmcEnergyExchangeSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcEnergyExchangeSolver>;

using DsmcEnergyExchangeSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcEnergyExchangeSolver>;

}
