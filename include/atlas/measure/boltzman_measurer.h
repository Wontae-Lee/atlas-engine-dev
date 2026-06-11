#pragma once

/**
 * @file boltzman_measurer.h
 * @brief Declares the BoltzmanMeasurer class for cell-wise kinetic-temperature measurement.
 *
 * `BoltzmanMeasurer` aggregates particle velocities per spatial cell and writes
 * thermodynamic field quantities into universe states.
 */

#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>

namespace atlas::system {

/**
 * @brief Measures cell-wise bulk velocity, thermal energy, particle count, and temperature.
 *
 * `BoltzmanMeasurer` is a concrete @ref Measurer implementation. It uses the
 * configured spatial hashing searcher to traverse particles grouped by universe
 * cell and computes the following values for each cell:
 *
 * - mean particle velocity, written to `UniverseBulkVelocityState<T>`,
 * - particle count, written to `UniverseNumberParticleState<T>`,
 * - summed squared velocity fluctuation, written to `UniverseThermalEnergyState<T>`,
 * - kinetic temperature, written to `UniverseTemperatureState<T>`.
 *
 * For a non-empty cell, the implementation first computes the mean velocity:
 *
 * @code
 * u = sum(v_i) / N
 * @endcode
 *
 * Then it computes the thermal-energy accumulator:
 *
 * @code
 * E = sum(|v_i - u|^2)
 * @endcode
 *
 * Finally, the cell temperature is computed as:
 *
 * @code
 * T_cell = E / (3 * atlas::boltzmann_constant * N)
 * @endcode
 *
 * Empty cells are reset to zero for bulk velocity, thermal energy, particle
 * count, and field temperature.
 *
 * If the configured measure mode is `MeasureModeType::Fluid` or
 * `MeasureModeType::All`, and `FluidTemperatureState<T>` exists, the measured
 * cell temperature is also copied back to each particle in the corresponding
 * cell. The constructor ensures that `FluidTemperatureState<T>` exists when a
 * fluid dependency is provided.
 *
 * @note The implementation name uses `Boltzman`, while the physical constant is
 *       referenced as `atlas::boltzmann_constant`.
 *
 * @tparam T Scalar type used by the simulation.
 */
template <typename T>
class BoltzmanMeasurer final : public Measurer<T> {
public:
    /**
     * @brief Fluent builder for constructing validated `BoltzmanMeasurer` instances.
     */
    class Builder;

public:
    /**
     * @brief Constructs an empty measurer.
     *
     * Dependencies are initialized to null-equivalent values and the measure mode
     * defaults to `MeasureModeType::All`. Calling @ref measure on an empty object
     * is safe because @ref Measurer::make_probe fails and the function returns
     * without launching kernels.
     */
    BoltzmanMeasurer() = default;

    /**
     * @brief Constructs a Boltzman measurer from simulation dependencies.
     *
     * The constructor stores the universe, fluid, and searcher through the
     * @ref Measurer base class and stores the selected measure mode.
     *
     * When a universe is provided, the constructor ensures that the following
     * universe states exist with `universe->number_of_cells()` elements:
     *
     * - `UniverseTemperatureState<T>`,
     * - `UniverseBulkVelocityState<T>`,
     * - `UniverseThermalEnergyState<T>`,
     * - `UniverseNumberParticleState<T>`.
     *
     * Existing states are left unchanged. Missing states are created.
     *
     * When a fluid is provided, the constructor ensures that
     * `FluidTemperatureState<T>` exists with `fluid->buffer_size()` elements.
     * Existing particle-temperature state is left unchanged.
     *
     * @param universe Universe receiving field measurement states.
     * @param fluid Fluid providing particle velocities and receiving optional
     *              particle temperatures.
     * @param searcher Spatial hashing searcher exposing cell-to-particle ranges.
     * @param measure_mode Controls whether measured field temperature is also
     *                     copied to particle temperature state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    BoltzmanMeasurer(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher,
                     MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    /**
     * @brief Destroys the measurer through the base interface.
     */
    ~BoltzmanMeasurer() override = default;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct a measurer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Executes the Boltzman measurement pass.
     *
     * The function first obtains a @ref Measurer::MeasurerProbe through
     * @ref Measurer::make_probe. If the probe cannot be built, the function
     * returns without modifying data.
     *
     * The first device pass iterates over every universe cell and computes:
     *
     * - mean velocity from valid particles in that cell,
     * - summed squared velocity fluctuation,
     * - particle count,
     * - field temperature.
     *
     * A particle index is considered valid only when it is within:
     *
     * @code
     * [0, probe.particle_count)
     * @endcode
     *
     * Empty cells are explicitly written with zero values.
     *
     * If particle-temperature output is available and the configured mode is
     * `MeasureModeType::Fluid` or `MeasureModeType::All`, a second device pass
     * copies each cell's measured field temperature into every valid particle in
     * that cell.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure() override;

    /**
     * @brief Executes the Boltzman measurement pass for a time-stepped caller.
     *
     * `BoltzmanMeasurer` computes instantaneous cell aggregates, so the time step
     * value is accepted for interface completeness and does not change the result.
     *
     * @param dt Simulation time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt) override;

    /**
     * @brief Returns the configured measurement mode.
     *
     * @return Current measure mode.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

private:
    /**
     * @brief Controls whether measured field temperature is copied back to particles.
     *
     * The field measurement pass always writes universe field states. The mode is
     * checked only for the optional particle-temperature write-back pass:
     *
     * - `MeasureModeType::Field`: field states only,
     * - `MeasureModeType::Fluid`: field states plus particle-temperature write-back,
     * - `MeasureModeType::All`: field states plus particle-temperature write-back.
     */
    MeasureModeType _measure_mode { MeasureModeType::All };
};

/**
 * @brief Fluent builder for `BoltzmanMeasurer`.
 *
 * The builder collects the required dependencies and a measurement mode, then
 * constructs a validated measurer.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * The builder rejects missing dependencies during @ref build. The default
 * builder measure mode is `MeasureModeType::Field`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
class BoltzmanMeasurer<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     *
     * The default measure mode is `MeasureModeType::Field`.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * The constructed measurer uses this universe as the target for field
     * measurement states.
     *
     * @param universe Host-side shared pointer to the target universe.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * The constructed measurer reads particle velocity from this fluid and may
     * write particle temperature to it depending on the configured measure mode.
     *
     * @param fluid Host-side shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * The constructed measurer uses the searcher's sorted indices and cell ranges
     * to aggregate particles by universe cell.
     *
     * @param searcher Host-side shared pointer to the spatial hashing searcher.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the measurement mode for the constructed measurer.
     *
     * The mode controls the optional particle-temperature write-back pass. Field
     * states are computed regardless of this mode.
     *
     * @param measure_mode Measurement mode to store in the constructed measurer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    /**
     * @brief Builds a validated measurer value.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null. The constructed measurer then creates any missing measurement
     * states required by @ref measure.
     *
     * @return Constructed measurer.
     *
     * @throw std::runtime_error If universe, fluid, or searcher is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE BoltzmanMeasurer<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared measurer.
     *
     * This function constructs a validated measurer using @ref build, then wraps
     * it in an `atlas::host_shared_ptr`.
     *
     * @return Host-side shared pointer to the constructed measurer.
     *
     * @throw std::runtime_error If universe, fluid, or searcher is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<BoltzmanMeasurer<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates that all required dependencies are configured.
     *
     * @throw std::runtime_error If the universe dependency is null.
     * @throw std::runtime_error If the fluid dependency is null.
     * @throw std::runtime_error If the searcher dependency is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Universe dependency collected by the builder.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency collected by the builder.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency collected by the builder.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Measurement mode collected by the builder.
     *
     * Defaults to `MeasureModeType::Field`.
     */
    MeasureModeType _measure_mode { MeasureModeType::Field };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::BoltzmanMeasurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurer = atlas::system::BoltzmanMeasurer<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::BoltzmanMeasurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurerHostPtr = atlas::host_shared_ptr<atlas::system::BoltzmanMeasurer<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::BoltzmanMeasurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurerDevicePtr = atlas::device_shared_ptr<atlas::system::BoltzmanMeasurer<T>>;

} // namespace atlas

#include <atlas/measure/boltzman_measurer.hpp>
