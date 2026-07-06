#pragma once

#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>

/**
 * @file boltzmann_measurer.h
 * @brief Derives per-cell (and optionally per-particle) translational
 *        temperature from the local spread of particle velocities, the
 *        standard DSMC/kinetic-theory macroscopic temperature estimator.
 *
 * @details
 * ### Derivation
 * By the equipartition theorem, a gas's translational temperature
 * relates to the mean-square velocity *fluctuation about the local mean
 * flow* (not the raw speed, which would conflate bulk motion with
 * thermal motion):
 * ```
 * (3/2) k_B T = (1/2) <|v - u|^2>
 * ```
 * where `u` is the local bulk (mean) velocity and the average is over
 * particles in the sampling region (here, one cell). Solving,
 * `T = <|v - u|^2> / (3 k_B)`. `measure_field()` computes exactly this
 * per cell: `bulk_velocity_ptr[cell] = mean(v)` over the cell's
 * particles, `thermal_energy_ptr[cell] = sum |v - u|^2`, and
 * `field_temperature_ptr[cell] = thermal_energy_ptr[cell] / (3 * k_B *
 * count)`. Note this formula has no explicit molecular-mass factor —
 * it implicitly assumes unit mass (or, equivalently, that velocities
 * are already mass-normalized); `MeasurerProbe` carries no
 * per-species/material data to weight a mixed-species population
 * correctly, so this measurer is appropriate for a single-species (or
 * otherwise mass-normalized) fluid.
 *
 * ### Operating principle
 * `measure()` always computes the per-cell field (`measure_field`);
 * if `measure_mode()` is `Fluid` or `All` *and* the fluid tracks
 * `FluidTemperatureState`, `assign_particle_temperature()` additionally
 * broadcasts each cell's field temperature back to every particle in
 * that cell (a coarse per-particle temperature — the *cell's* estimate,
 * not a genuinely per-particle quantity, since temperature is only
 * meaningful as a statistical property of a population).
 */

namespace atlas {

/**
 * @brief Equipartition-theorem translational temperature estimator from
 *        per-cell velocity spread. See this file's top-of-file
 *        documentation for the derivation and the implicit unit-mass
 *        assumption.
 */
class BoltzmannMeasurer final : public Measurer {
public:
    class Builder;

public:
    BoltzmannMeasurer() = default;

    ATLAS_HOST BoltzmannMeasurer(UniverseHostPtr universe,
                                 FluidHostPtr fluid,
                                 SearcherHostPtr searcher,
                                 MeasureModeType measure_mode = MeasureModeType::all) noexcept;

    ~BoltzmannMeasurer() override = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    /** @brief Rebuilds the probe and runs `measure_field()` (and
     *  `assign_particle_temperature()` per `measure_mode()`); no-op if
     *  the probe can't be built. */
    ATLAS_HOST void
    measure() override;

    /** @brief `measure()` — this measurer's computation does not depend
     *  on `dt`. */
    ATLAS_HOST void
    measure(float dt) override;

    ATLAS_HOST ATLAS_NODISCARD MeasureModeType
    measure_mode() const noexcept override;

public:
    /** @brief Per-cell bulk velocity/thermal energy/temperature
     *  computation; see this file's top-of-file Derivation. */
    ATLAS_HOST void
    measure_field();

    /** @brief Broadcasts each cell's `field_temperature_ptr` to every
     *  particle in that cell's `particle_temperature_ptr`. */
    ATLAS_HOST void
    assign_particle_temperature();

private:
    MeasureModeType _measure_mode { MeasureModeType::all };
};

/**
 * @brief Fluent builder for `BoltzmannMeasurer`. Validation requires
 *        non-null `_universe`/`_fluid`/`_searcher`.
 */
class BoltzmannMeasurer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_HOST ATLAS_NODISCARD BoltzmannMeasurer
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<BoltzmannMeasurer>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    MeasureModeType _measure_mode { MeasureModeType::field };
};

using BoltzmannMeasurerHostPtr = atlas::host_shared_ptr<BoltzmannMeasurer>;

using BoltzmannMeasurerDevicePtr = atlas::device_shared_ptr<BoltzmannMeasurer>;

}
