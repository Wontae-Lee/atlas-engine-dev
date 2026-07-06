#pragma once

#include <atlas/core/macros.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>

/**
 * @file sph_solver.h
 * @brief The (weakly-compressible) SPH continuum solver: per particle,
 *        estimates local density from neighbors, derives pressure from a
 *        linear equation of state, and integrates pressure + viscous
 *        forces into a velocity update.
 *
 * @details
 * ### Background
 * This solver follows the same simplified, real-time-oriented SPH force
 * model as `StandardSphKernel` (Müller, Charypar & Gross 2003) rather
 * than a fully symmetric, astrophysically-rigorous SPH discretization,
 * even when a different kernel shape (cubic spline / Wendland) is
 * selected: a *linear* (Hookean) equation of state
 * `P = k * (rho - rho0)` relates local density directly to pressure
 * (`rho0` the material's rest density, `k` its `pressure_coefficient`),
 * rather than the stiffer Tait-type EOS common in "weakly compressible"
 * SPH proper; pressure and viscous forces are then assembled from the
 * selected `SphKernel`'s `pressure_gradient`/`viscosity_laplacian`. This
 * gives Atlas an SPH solver simple and cheap enough to run alongside
 * DSMC while remaining physically consistent
 * — the linear EOS and force assembly are exactly Müller et al.'s
 * construction, generalized here to accept any of the three kernel
 * shapes.
 *
 * ### Operating principle — the per-step pipeline (`solve(dt)`)
 * 1. `initialize_context`/`prepare_fields`: ensure per-cell universe
 *    states and per-particle scratch buffers (`_density`/`_pressure`/
 *    `_acceleration`) are sized, and rebuild the searcher's spatial
 *    partition + neighbor lists (needed since SPH, unlike DSMC, sums
 *    over *all* neighbors within the kernel support, not just cell-mates
 *    — see `sph_probe.h`'s `neighbor_offsets_ptr`/`neighbor_indices_ptr`).
 * 2. `make_probe`: rebuild the `SphProbe` view (see that file).
 * 3. `update` -> `estimate_density`: per particle, sums
 *    `density = self-term + sum_j m_j * W(r_ij, h)` over neighbors
 *    within the kernel's support (`density_weight(0, h)` is the
 *    particle's own contribution to its density, a standard SPH
 *    self-term), then evaluates the linear EOS `pressure = k *
 *    (density - rho0)`. Falls back to `rho0` if the neighbor sum comes
 *    out non-positive (an isolated particle with no self-term
 *    contribution, degenerate).
 * 4. `accelerate(dt)`: per particle, accumulates over the same neighbor
 *    list:
 *      - **Pressure force**: `-grad(W) * m_j * (P_i + P_j) / (2 * rho_j)`
 *        — the symmetrized pressure-gradient SPH force (using the
 *        *average* of the two particles' pressures divides by only the
 *        neighbor's density rather than the fully symmetric
 *        `P_i/rho_i^2 + P_j/rho_j^2` form; this is Müller et al.'s
 *        simplified variant, cheaper and still momentum-antisymmetric
 *        between a pair since `grad(W)` itself is antisymmetric in
 *        `delta`).
 *      - **Viscous force** (only if the material has positive
 *        `dynamic_viscosity`): `+ (v_j - v_i) * mu * m_j *
 *        laplacian(W) / rho_j` — a Laplacian-smoothed relaxation of
 *        each particle's velocity toward its neighbors', the standard
 *        SPH viscosity term.
 *    then a semi-implicit Euler velocity update
 *    `v += acceleration * dt` is applied in place. A second pass
 *    averages each cell's particle accelerations into
 *    `UniverseFieldForceState` (`field_force_ptr`), a coarse per-cell
 *    force field other systems (e.g. rendering/diagnostics, or as an
 *    external-force input elsewhere) can sample without re-deriving
 *    per-particle forces.
 * Note `SphSolver` does **not** integrate position — only velocity; a
 * separate position-integration step (e.g. `System::time_integration`)
 * applies the updated velocities to particle positions.
 *
 * ### References
 * - M. Müller, D. Charypar, and M. Gross, "Particle-Based Fluid
 *   Simulation for Interactive Applications," ACM SIGGRAPH/Eurographics
 *   Symposium on Computer Animation, 2003. (the linear EOS and
 *   pressure/viscosity force assembly this solver implements)
 */

namespace atlas {

/**
 * @brief Weakly-compressible SPH continuum solver (linear EOS,
 *        neighbor-summed pressure/viscosity forces). See this file's
 *        top-of-file documentation for the full per-step pipeline.
 */
class SphSolver final : public Solver {
public:
    using SphSolverProbe = atlas::SphProbe;

    class Builder;

    SphSolver() = default;

    ATLAS_HOST SphSolver(UniverseHostPtr universe,
                         FluidHostPtr fluid,
                         SearcherHostPtr searcher,
                         SphKernelType kernel_type = SphKernelType::standard) noexcept;

    ~SphSolver() override = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_NODISCARD SphKernelType
    kernel_type() const noexcept;

    /** @brief Full per-step pipeline (see this file's top-of-file
     *  documentation): rebuild neighbor lists/probe, estimate density
     *  and pressure, accumulate and apply forces. Throws if
     *  `dt <= 0` once the solver has valid context; no-op (silent
     *  return) if universe/fluid/searcher/particle state is not ready. */
    ATLAS_HOST void
    solve(float dt) override;

    /** @brief Not implemented for `SphSolver` (no-op); the base's
     *  per-cell solver-allocation entry point (see `solver.h`) is
     *  unused on the SPH path. */
    ATLAS_HOST void
    solve(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    /** @brief Allocates/resizes the per-cell `UniverseNumberParticleState`/
     *  `UniverseFieldForceState` universe states this solver writes. */
    ATLAS_HOST void
    ensure_states();

    /** @brief Validates universe/fluid/searcher/particle-state
     *  preconditions, ensures states, and rebuilds the searcher; resets
     *  scratch buffers and returns `false` if any precondition fails. */
    ATLAS_HOST bool
    initialize_context() noexcept;

    /** @brief Rebuilds `_probe` from current universe/fluid/searcher/
     *  kernel state (`detail::SphProbeBuilder::make`). */
    ATLAS_HOST ATLAS_NODISCARD bool
    make_probe() noexcept;

    /** @brief Resizes `_density`/`_pressure`/`_acceleration` to the
     *  current particle count; `false` (after clearing/resetting) if
     *  there are zero particles. */
    ATLAS_HOST bool
    prepare_fields();

    /** @brief Zeros the per-cell `UniverseNumberParticleState`/
     *  `UniverseFieldForceState` universe states. */
    ATLAS_HOST void
    reset_fields();

    /** @brief The material's configured `rest_density` (`rho0` in the
     *  linear EOS), or `1.0` if unset/non-positive. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    rest_density(const MaterialProperties& property) noexcept;

    /** @brief The material's configured `pressure_coefficient` (`k` in
     *  the linear EOS `P = k*(rho - rho0)`), or `0` if unset. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    pressure_coefficient(const MaterialProperties& property) noexcept;

    /** @brief Per-particle neighbor-summed density + linear-EOS pressure
     *  (step 3 of this file's top-of-file pipeline). */
    ATLAS_HOST void
    estimate_density();

    /** @brief Per-cell particle counts into
     *  `UniverseNumberParticleState`, for diagnostics/other systems. */
    ATLAS_HOST void
    count_particles();

    /** @brief Pressure + viscous force accumulation and the velocity
     *  update, plus the per-cell averaged field-force pass (step 4 of
     *  this file's top-of-file pipeline). */
    ATLAS_HOST void
    accelerate(float dt);

    /** @brief `estimate_density()` then `count_particles()`. */
    ATLAS_HOST void
    update();

private:
    SphSolverProbe _probe {};

    SphKernel _kernel {};

    DeviceBuffer<float> _density {};

    DeviceBuffer<float> _pressure {};

    DeviceBuffer<Vector3> _acceleration {};
};

/**
 * @brief Fluent builder for `SphSolver`. Validation (`validate()`, run
 *        by `build()`/`make_host_shared()`) requires non-null
 *        `_universe`/`_fluid`/`_searcher`.
 */
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

    ATLAS_HOST ATLAS_NODISCARD SphSolver
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<SphSolver>
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
