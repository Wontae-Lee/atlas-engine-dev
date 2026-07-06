#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer_probe.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

/**
 * @file measurer.h
 * @brief Host-only base interface for diagnostic field/particle
 *        measurement: derives macroscopic quantities (temperature, bulk
 *        velocity, ...) from a fluid's microscopic particle state,
 *        without feeding back into the simulated dynamics.
 *
 * @details
 * Measurers are read-only observers of the simulation: unlike
 * `Solver`/`Collider`/`Sink`/`Source`, nothing here changes particle
 * positions, velocities, or counts — a `Measurer` only writes derived
 * diagnostic state (per-cell universe states like
 * `UniverseTemperatureState`, or per-particle fluid states like
 * `FluidTemperatureState`) for other systems, logging, or visualization
 * to consume. `measure_mode()` (see `MeasureModeType`) lets a concrete
 * measurer report whether it writes per-cell field data, per-particle
 * data, or both, so a caller can skip work it doesn't need.
 */

namespace atlas {

/**
 * @brief What granularity of diagnostic data a `Measurer::measure()`
 *        call writes.
 */
enum class MeasureModeType : int {

    /** Writes only per-cell (universe) field state. */
    field,

    /** Writes only per-particle (fluid) state. */
    fluid,

    /** Writes both. */
    all
};

/**
 * @brief Base interface for read-only diagnostic measurement over a
 *        `Fluid`/`Universe`/`Searcher` triple. See this file's
 *        top-of-file documentation.
 */
class Measurer {
public:
    Measurer() = default;

    ATLAS_HOST Measurer(UniverseHostPtr universe,
                        FluidHostPtr fluid,
                        SearcherHostPtr searcher) noexcept;

    virtual ~Measurer() = default;

    Measurer(const Measurer&) = default;
    Measurer&
    operator=(const Measurer&)
        = default;
    Measurer(Measurer&&) noexcept = default;
    Measurer&
    operator=(Measurer&&) noexcept = default;

    /** @brief Runs one measurement pass, writing whatever diagnostic
     *  state `measure_mode()` reports this measurer produces. */
    ATLAS_HOST virtual void
    measure()
        = 0;

    /** @brief `dt`-aware overload for measurers whose diagnostic
     *  computation depends on the timestep (e.g. `VolumeMeasurer`
     *  advancing its own units before measuring); the base
     *  implementation simply forwards to `measure()`. */
    ATLAS_HOST virtual void
    measure(float dt);

    ATLAS_HOST ATLAS_NODISCARD virtual MeasureModeType
    measure_mode() const noexcept = 0;

    /** @brief Rebuilds `_probe` from the current universe/fluid/
     *  searcher state; `false` if any is null or the required universe/
     *  fluid states/searcher arrays are missing. */
    ATLAS_HOST ATLAS_NODISCARD bool
    make_probe() noexcept;

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    MeasurerProbe _probe {};
};

using MeasurerHostPtr = atlas::host_shared_ptr<Measurer>;

using MeasurerDevicePtr = atlas::device_shared_ptr<Measurer>;

}
