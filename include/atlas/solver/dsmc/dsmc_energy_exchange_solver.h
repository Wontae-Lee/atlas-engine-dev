#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

#include <cmath>

/**
 * @file dsmc_energy_exchange_solver.h
 * @brief `DsmcSolver` extended with Larsen-Borgnakke rotational/
 *        vibrational energy exchange: on each accepted collision, some
 *        translational kinetic energy may relax into (or out of)
 *        internal energy before the elastic scattering kernel runs.
 *
 * @details
 * ### Background
 * Plain `DsmcSolver`/`DsmcKernel::operator()` model collisions as purely
 * elastic: translational kinetic energy is conserved and only its
 * direction is redistributed. Real polyatomic molecules also exchange
 * energy with internal (rotational, vibrational) modes on collision, and
 * that exchange happens with finite probability, not on every collision
 * — a diatomic gas needs, on average, `Zr` (the rotational collision
 * number) collisions before a given molecule's rotational mode
 * equilibrates with translation, and `Zv >> Zr` for vibration. The
 * Larsen-Borgnakke (LB) statistical model (Larsen & Borgnakke, 1975) is
 * the standard DSMC treatment: on each collision, redistribute energy
 * between translation and each internal mode with probability `1/Z`
 * (`Z` = that mode's relaxation collision number), and when relaxation
 * *does* happen, redraw the post-collision split from the equilibrium
 * distribution appropriate to the modes' combined degrees of freedom —
 * exactly the same “redistribute total energy among quadratic degrees of
 * freedom” idea as `MaxwellianSurfaceInteraction`'s internal-energy
 * sampling (see that file's Derivation D3), except here between *two
 * colliding particles'* translational and internal energy pools instead
 * of a particle and a wall.
 *
 * `Zr`/`Zv` are not constants in a real gas — they rise sharply at low
 * collision energy (a "cold" collision is unlikely to excite rotation/
 * vibration at all) — so `rotational_relaxation_probability`/
 * `vibrational_relaxation_probability` use temperature-dependent
 * formulas (Parker's formula for `Zr(T)`, extended in Bird's DSMC;
 * a Millikan-White-derived formula for `Zv(T)`) rather than the
 * material's optional constant `*_relaxation_probability` fallback.
 *
 * ### Operating principle
 * `collide_indexed_pair` mirrors `DsmcSolver`'s NTC accept/reject test
 * exactly (same `sigma_g`/`max_sigma_g` bookkeeping — see
 * `dsmc_solver.h`), but on acceptance calls `exchange_internal_energy`
 * *before* `probe.kernel(...)`'s elastic scattering, then
 * `rescale_relative_velocity` to make the post-scattering relative speed
 * consistent with whatever translational energy survived the exchange.
 *
 * `exchange_internal_energy`:
 * 1. Pools the pair's relative translational kinetic energy
 *    `e_dispose = 0.5 * mu * g^2` (`mu` = reduced mass, `g` = relative
 *    speed) — the energy available to redistribute.
 * 2. Calls `exchange_particle_internal_energy` once per particle. Each
 *    call, per mode (rotational then vibrational):
 *      a. Computes that mode's relaxation probability at the
 *         *candidate* collision energy (`e_dispose + <that particle's
 *         current mode energy>`).
 *      b. Draws a hashed uniform; on success (relaxation happens this
 *         collision), adds the particle's current mode energy into the
 *         shared pool `e_dispose`, redraws the mode's *new* energy from
 *         the pool, then subtracts the new draw back out of the pool —
 *         so `e_dispose` always reflects exactly the energy not yet
 *         assigned to any mode.
 * 3. Whatever remains in `e_dispose` after both particles' modes have
 *    had a chance to exchange becomes the pair's *new* relative
 *    translational kinetic energy, which `rescale_relative_velocity`
 *    then imposes by rescaling `relative` to the speed that energy
 *    implies (preserving direction and center-of-mass velocity), before
 *    `probe.kernel(...)` re-scatters that (rescaled) relative velocity's
 *    *direction* per the configured collision model.
 *
 * ### Derivations
 * **New-energy sampling given a relaxation event.** With `dof` degrees
 * of freedom exchanging out of a pool `E`:
 *   - `dof == 2` (a single rotational mode, or a harmonic-oscillator
 *     vibrational mode with `omega`-adjusted exponent): drawn as
 *     `E_new = (1 - U^(1/exponent)) * E`, `exponent = 2.5 - omega`,
 *     `U ~ Uniform(0,1)` — the standard closed-form LB two-body
 *     redistribution: this is the fraction of `E` assigned to a 2-dof
 *     mode when the *other* share (the remaining translational +
 *     other-mode dof) behaves like a `(2*exponent)`-dof continuum, and
 *     is exact (needs no rejection) because the 2-dof share has a simple
 *     power-law marginal.
 *   - `dof != 2`: `sample_bl(a1, a2, ...)` (an acceptance-rejection Beta-
 *     distribution-shaped sampler: draws `x ~ Uniform(0,1)`, accepts
 *     with probability `(x*(a1+a2)/a1)^a1 * ((1-x)*(a1+a2)/a2)^a2`, the
 *     peak-normalized Beta(a1+1, a2+1)-shaped density — the same
 *     acceptance-rejection idea as `maxwellian_surface_interaction.h`'s
 *     Derivation D3, generalized to two competing power-law exponents
 *     `a1 = dof/2 - 1` (the relaxing mode) and `a2 = 1.5 - omega` (the
 *     remaining share)) gives the fraction of `E` assigned to the mode;
 *     `E_new = E * sample_bl(...)`.
 *
 * ### References
 * - C. Borgnakke and P. S. Larsen, "Statistical collision model for
 *   Monte Carlo simulation of polyatomic gas mixture," Journal of
 *   Computational Physics, 18(4), 1975, pp. 405-420. (the Larsen-
 *   Borgnakke model this solver implements)
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994, ch. 5. (LB redistribution
 *   formulas for `dof == 2` and general `dof`, and the extension of
 *   Parker's formula for temperature-dependent `Zr`)
 * - J. G. Parker, "Rotational relaxation in diatomic gases," Physics of
 *   Fluids 2(4), 1959, pp. 449-462. (temperature-dependent rotational
 *   collision number, extended here in `rotational_relaxation_probability`)
 * - R. C. Millikan and D. R. White, "Systematics of vibrational
 *   relaxation," Journal of Chemical Physics 39(12), 1963,
 *   pp. 3209-3213. (basis for the temperature-dependent vibrational
 *   collision number in `vibrational_relaxation_probability`)
 */

namespace atlas {

/**
 * @brief `DsmcSolver` with Larsen-Borgnakke rotational/vibrational
 *        energy exchange on each accepted collision. See this file's
 *        top-of-file documentation for the model and its derivation.
 */
class DsmcEnergyExchangeSolver final : public DsmcSolver {
public:
    using Base  = DsmcSolver;
    using Probe = Base::Probe;
    using Base::Base;

    class Builder;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    /** @brief NTC selection plus Larsen-Borgnakke energy exchange for
     *  every cell; overrides `DsmcSolver::apply_collision`. */
    ATLAS_HOST void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    /** @brief Resolves local slot indices to particle indices and
     *  forwards to `collide_indexed_pair` (energy-exchange counterpart
     *  of `DsmcSolver::collide_pair`). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 float max_sigma_g) noexcept;

    /** @brief NTC accept/reject test identical to
     *  `DsmcSolver::collide_indexed_pair`, but on acceptance runs
     *  `exchange_internal_energy` before the kernel's elastic scattering
     *  and `rescale_relative_velocity` after. See this file's
     *  top-of-file documentation for the full sequence. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         int local_collision,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         float max_sigma_g) noexcept;

    /** @brief Hashed uniform draw in `[0,1)` keyed by
     *  `(cell, local_collision, seed, salt)` — the shared RNG primitive
     *  every sampler in this file is built from. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    sample_unit(int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    /**
     * @brief Acceptance-rejection sample of a Beta(`exp_1`+1, `exp_2`+1)-
     *        shaped fraction in `[0,1]`: the general (non-`dof==2`)
     *        Larsen-Borgnakke redistribution fraction. See this file's
     *        Derivations section. Falls back to `0.5` after 32 failed
     *        attempts (astronomically unlikely for the exponent ranges
     *        this solver uses); returns `0` if either exponent is
     *        non-positive.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    sample_bl(float exp_1, float exp_2, int cell, int local_collision, std::uint64_t seed, std::uint64_t salt) noexcept;

    /**
     * @brief Temperature-dependent rotational relaxation probability
     *        `1/Zr(T_c)` (Parker's formula, extended per Bird 1994) from
     *        `material`'s `rotational_relaxation_c1/c2/c3` fit
     *        constants and the pair's collision temperature implied by
     *        `collision_energy`; falls back to the material's constant
     *        `rotational_relaxation_probability` if the fit constants
     *        are not set. Returns `0` if the material has no rotational
     *        dof.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    rotational_relaxation_probability(const MaterialProperties& material,
                                      float collision_energy,
                                      float omega) noexcept;

    /** @brief Analogous temperature-dependent vibrational relaxation
     *  probability `1/Zv(T_c)`, derived from a Millikan-White-style fit
     *  (`vibrational_relaxation_c1/c2`); same fallback/zero-dof
     *  behavior as `rotational_relaxation_probability`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    vibrational_relaxation_probability(const MaterialProperties& material,
                                       float collision_energy,
                                       float omega) noexcept;

    /**
     * @brief Pools the pair's relative translational kinetic energy and
     *        runs `exchange_particle_internal_energy` for both
     *        particles' rotational/vibrational modes in turn.
     * @return The energy remaining in the pool after both particles'
     *         modes have had a chance to relax — the pair's new
     *         relative translational kinetic energy. `0` if the pair's
     *         `DsmcPairParameters` are invalid (see
     *         `DsmcKernel::pair_parameters`); the raw pooled energy
     *         (unmodified) if the fluid does not track internal energy
     *         at all.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    exchange_internal_energy(const Probe& probe,
                             int cell,
                             int local_collision,
                             int particle_i,
                             int particle_j,
                             std::size_t species_i,
                             std::size_t species_j,
                             float relative_speed_squared) noexcept;

    /**
     * @brief Rotational-then-vibrational relaxation for one particle:
     *        per mode, draws whether relaxation happens this collision
     *        (`rotational_relaxation_probability`/
     *        `vibrational_relaxation_probability`), and if so moves that
     *        mode's current energy into `e_dispose` and redraws its new
     *        energy from the pool (`dof == 2` closed form or
     *        `sample_bl` for `dof > 2`; see this file's Derivations).
     *        Writes the particle's updated `FluidInternalEnergy` back to
     *        `probe.internal_energy_ptr`, with `energy.translational`
     *        set to whatever remains in `e_dispose` once both modes have
     *        had their turn.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    exchange_particle_internal_energy(const Probe& probe,
                                      int cell,
                                      int local_collision,
                                      int particle,
                                      const MaterialProperties& material,
                                      float omega,
                                      std::uint64_t salt_base,
                                      float& e_dispose) noexcept;

    /**
     * @brief Rescales `lhs_velocity`/`rhs_velocity`'s relative-velocity
     *        magnitude (about their shared center-of-mass velocity) so
     *        the pair's relative translational kinetic energy equals
     *        `translational_energy`, preserving direction and momentum.
     *        Applied *before* `DsmcKernel::operator()` reorients that
     *        (now correctly-scaled) relative velocity's direction. No-op
     *        for degenerate input (non-positive masses/energy or zero
     *        relative speed).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    rescale_relative_velocity(Vector3& lhs_velocity,
                              Vector3& rhs_velocity,
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

/**
 * @brief Fluent builder for `DsmcEnergyExchangeSolver`; same required
 *        fields (universe/fluid/searcher) and validation as
 *        `DsmcSimpleSolver::Builder`.
 */
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

    ATLAS_HOST ATLAS_NODISCARD DsmcEnergyExchangeSolver
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<DsmcEnergyExchangeSolver>
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

    Vector3 lhs_velocity               = probe.velocity_ptr[particle_i];
    Vector3 rhs_velocity               = probe.velocity_ptr[particle_j];
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
DsmcEnergyExchangeSolver::rescale_relative_velocity(Vector3& lhs_velocity,
                                                    Vector3& rhs_velocity,
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

    const Vector3 relative = lhs_velocity - rhs_velocity;
    const float speed      = relative.length();
    if (!(speed > 0.0f)) {
        return;
    }

    const float target_speed         = atlas::sqrt_nonnegative(2.0f * translational_energy / reduced_mass);
    const Vector3 scattered_relative = relative * (target_speed / speed);
    const Vector3 center             = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

using DsmcEnergyExchangeSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcEnergyExchangeSolver>;

using DsmcEnergyExchangeSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcEnergyExchangeSolver>;

}
