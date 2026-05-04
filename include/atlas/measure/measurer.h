#pragma once

/**
 * @file measurer.h
 * @brief Declares the base Measurer interface used to compute universe-level measurements.
 *
 * A measurer reads particle-level fluid data through a spatial searcher and
 * writes aggregated measurement values into universe states.
 */

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Identifies the category of data produced by a measurer.
 *
 * Derived measurers return this value from @ref Measurer::measure_mode to
 * describe whether they update field-level data, fluid-level data, or both.
 */
enum class MeasureModeType : int {
    /**
     * @brief The measurer primarily updates universe field/cell measurements.
     */
    Field,

    /**
     * @brief The measurer primarily updates fluid/particle measurements.
     */
    Fluid,

    /**
     * @brief The measurer updates both field-level and fluid-level measurements.
     */
    All
};

/**
 * @brief Abstract base class for measurement passes over a fluid/universe pair.
 *
 * `Measurer` stores the shared dependencies required to aggregate particle data
 * into universe measurement states:
 *
 * - a universe containing output measurement states,
 * - a fluid containing particle data,
 * - a spatial hashing searcher exposing cell-to-particle ranges.
 *
 * The base class does not implement the measurement algorithm itself. Derived
 * classes implement @ref measure and may use @ref make_probe to obtain raw
 * pointer views over the common input/output buffers required by measurement
 * kernels.
 *
 * The common probe resolves the following required universe states:
 *
 * - `UniverseTemperatureState<T>`,
 * - `UniverseBulkVelocityState<T>`,
 * - `UniverseThermalEnergyState<T>`,
 * - `UniverseNumberParticleState<T>`,
 *
 * and the required fluid state:
 *
 * - `FluidVelocityState<T>`.
 *
 * `FluidTemperatureState<T>` is optional and is exposed through the probe only
 * when present.
 *
 * @tparam T Scalar type used by the simulation, for example `float` or `double`.
 */
template <typename T>
class Measurer {
public:
    /**
     * @brief Raw-pointer view over common data used by measurement kernels.
     *
     * `MeasurerProbe` is populated by @ref make_probe. It groups output universe
     * measurement buffers, input fluid particle buffers, and searcher cell-range
     * data into one compact object suitable for capture by device lambdas.
     *
     * Required output buffers are:
     *
     * - field temperature,
     * - bulk velocity,
     * - thermal energy,
     * - number of particles.
     *
     * Required input/search buffers are:
     *
     * - particle velocity,
     * - sorted particle indices,
     * - cell start offsets,
     * - cell end offsets.
     *
     * Particle temperature is optional and may remain `nullptr`.
     */
    struct MeasurerProbe {
        /**
         * @brief Raw pointer to per-cell field temperature output.
         *
         * Points to `UniverseTemperatureState<T>::data()`.
         */
        T* field_temperature_ptr {};

        /**
         * @brief Raw pointer to per-cell bulk velocity output.
         *
         * Points to `UniverseBulkVelocityState<T>::data()`.
         */
        Vector3<T>* bulk_velocity_ptr {};

        /**
         * @brief Raw pointer to per-cell thermal energy output.
         *
         * Points to `UniverseThermalEnergyState<T>::data()`.
         */
        T* thermal_energy_ptr {};

        /**
         * @brief Raw pointer to per-cell particle-count output.
         *
         * Points to `UniverseNumberParticleState<T>::data()`.
         */
        T* number_particle_ptr {};

        /**
         * @brief Raw pointer to per-particle velocity input.
         *
         * Points to `FluidVelocityState<T>::data()`. This pointer is required
         * for a valid probe.
         */
        const Vector3<T>* velocity_ptr {};

        /**
         * @brief Raw pointer to optional per-particle temperature data.
         *
         * Points to `FluidTemperatureState<T>::data()` when that state exists.
         * Remains `nullptr` when the fluid does not expose particle temperature.
         */
        T* particle_temperature_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * For each cell, the range
         * `[cell_start_ptr[cell], cell_end_ptr[cell])` indexes into this array.
         */
        const int* indices_ptr {};

        /**
         * @brief Raw pointer to the first sorted index for each cell.
         */
        const int* cell_start_ptr {};

        /**
         * @brief Raw pointer to one-past-the-last sorted index for each cell.
         */
        const int* cell_end_ptr {};

        /**
         * @brief Number of particles reported by the fluid.
         *
         * A value of zero is accepted as valid by @ref make_probe.
         */
        int particle_count {};

        /**
         * @brief Number of cells reported by the universe.
         *
         * Must be greater than zero for @ref make_probe to succeed.
         */
        int num_of_cells {};
    };

    /**
     * @brief Constructs an empty measurer.
     *
     * Universe, fluid, and searcher dependencies are initialized to null-equivalent
     * values. Calling @ref make_probe on such an object returns `false`.
     */
    Measurer() = default;

    /**
     * @brief Constructs a measurer from its shared simulation dependencies.
     *
     * The constructor stores the provided host-side shared pointers by move. It
     * performs no state validation; required states are checked later by
     * @ref make_probe.
     *
     * @param universe Universe containing measurement output states.
     * @param fluid Fluid containing particle data used as measurement input.
     * @param searcher Spatial searcher exposing cell-to-particle ranges.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Measurer(UniverseHostPtr<T> universe,
             FluidHostPtr<T> fluid,
             SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Destroys the measurer through the base interface.
     */
    virtual ~Measurer() = default;

    /**
     * @brief Executes the concrete measurement pass.
     *
     * Derived classes implement this function to compute measurement values from
     * fluid particle data and write the result into universe states. Implementations
     * commonly call @ref make_probe before launching their measurement kernel.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure()
        = 0;

    /**
     * @brief Returns the measurement category implemented by the concrete measurer.
     *
     * This value allows callers to distinguish field-only, fluid-only, and mixed
     * measurement passes.
     *
     * @return Measurement mode implemented by the derived measurer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual MeasureModeType
    measure_mode() const noexcept = 0;

    /**
     * @brief Populates a raw-pointer probe for measurement kernels.
     *
     * This function resolves the configured universe, fluid, and searcher into
     * raw pointer views over common measurement data.
     *
     * Required dependencies:
     *
     * - non-null universe,
     * - non-null fluid,
     * - non-null spatial searcher.
     *
     * Required universe states:
     *
     * - `UniverseTemperatureState<T>`,
     * - `UniverseBulkVelocityState<T>`,
     * - `UniverseThermalEnergyState<T>`,
     * - `UniverseNumberParticleState<T>`.
     *
     * Required fluid state:
     *
     * - `FluidVelocityState<T>`.
     *
     * Optional fluid state:
     *
     * - `FluidTemperatureState<T>`.
     *
     * The probe succeeds only when particle count is non-negative, universe cell
     * count is positive, and the searcher exposes non-null `indices`,
     * `cell_start`, and `cell_end` arrays.
     *
     * @param probe Output probe populated with raw pointers and scalar metadata.
     *
     * @retval true Required dependencies, states, and searcher buffers were found.
     * @retval false A required dependency, state, or searcher buffer was missing,
     *               or the universe cell count was not positive.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe(MeasurerProbe& probe) noexcept;

protected:
    /**
     * @brief Universe dependency containing measurement output states.
     *
     * May be null. Required by @ref make_probe.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency containing particle input states.
     *
     * May be null. Required by @ref make_probe.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial searcher dependency containing cell-to-particle ranges.
     *
     * May be null. Required by @ref make_probe.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::MeasureModeType`.
 */
using MeasureModeType = atlas::system::MeasureModeType;

/**
 * @brief Convenience alias for `atlas::system::Measurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using Measurer = atlas::system::Measurer<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::Measurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using MeasurerHostPtr = atlas::host_shared_ptr<atlas::system::Measurer<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::Measurer`.
 *
 * @tparam T Scalar type used by the measurer.
 */
template <typename T>
using MeasurerDevicePtr = atlas::device_shared_ptr<atlas::system::Measurer<T>>;

} // namespace atlas

#include <atlas/measure/measurer.hpp>