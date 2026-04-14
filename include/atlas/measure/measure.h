#pragma once

/**
 * @file measure.h
 * @brief Declares the abstract base interface for all measurement components in Atlas.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::MeasureModeType, which specifies how measurements are applied,
 * - @ref atlas::system::Measure, the polymorphic base class for all measurement systems.
 *
 * ## Conceptual overview
 * A Measure represents a computation that:
 * - observes the simulation state,
 * - derives diagnostic or physical quantities,
 * - optionally writes results to domain fields, particle buffers, or external outputs.
 *
 * Measurements are **read-only with respect to simulation evolution** (they do not
 * advance the simulation state), but they may produce side effects such as logging,
 * accumulation, or field updates.
 *
 * ## Execution context
 * All measurement operations are executed using runtime probes:
 * - @ref DomainDeviceProbe → grid / field data
 * - @ref SpatialHashingProbe → spatial neighborhood queries
 * - @ref FluidDeviceProbe → particle state
 *
 * These probes abstract host/device execution and allow measurement logic to run
 * efficiently across CPU and GPU backends.
 *
 * ## Design goals
 * - Provide a unified interface for all measurement types
 * - Enable polymorphic composition of measurement systems
 * - Decouple measurement logic from solver and simulation core
 *
 * ---
 */

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

/**
 * @brief Specifies how a measurement should be applied within the simulation.
 *
 * @details
 * This enumeration controls the **scope and target** of a measurement.
 *
 * It allows measurement implementations to:
 * - restrict computation to specific data domains,
 * - optimize execution by skipping unnecessary work,
 * - differentiate between field-based and particle-based evaluations.
 *
 * ## Modes
 * - Field:
 *   Measurement operates primarily on domain/grid data.
 *   Example: temperature field reconstruction.
 *
 * - Fluid:
 *   Measurement operates on particle/fluid data.
 *   Example: velocity statistics or particle-based temperature.
 *
 * - All:
 *   Measurement uses both domain and particle data.
 *   This is the default and most general mode.
 */
enum class MeasureModeType : int {
    Field, ///< Measure only domain/field data.
    Fluid, ///< Measure only particle/fluid data.
    All    ///< Measure both field and particle data.
};

/**
 * @brief Abstract base class for all measurement components.
 *
 * @details
 * `Measure<T>` defines the minimal interface required for any measurement
 * system in Atlas. It is designed for **runtime polymorphism** and integration
 * into higher-level orchestration pipelines.
 *
 * ## Responsibilities
 * Derived classes must:
 * - implement @ref measure to perform the measurement,
 * - implement @ref measure_mode to report their operational mode.
 *
 * ## Measurement contract
 * The @ref measure function:
 * - reads simulation state from probes,
 * - computes derived quantities,
 * - may store or output results (implementation-defined).
 *
 * It must not:
 * - mutate core simulation state in a way that affects solver correctness.
 *
 * ## Typical derived classes
 * - Thermometers (e.g., average/variance temperature)
 * - Statistical analyzers
 * - Field reconstruction operators
 * - Diagnostics and logging systems
 *
 * ## Polymorphic usage
 * Instances are typically used via:
 * - @ref MeasureHostPtr for host-side orchestration,
 * - @ref MeasureDevicePtr for device-side execution (if supported).
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class Measure {
public:
    /**
     * @brief Default constructor.
     */
    Measure() = default;

    /**
     * @brief Virtual destructor.
     */
    virtual ~Measure() = default;

    /**
     * @brief Perform a measurement using the current simulation state.
     *
     * @details
     * This pure virtual function must be implemented by derived classes.
     *
     * ## Inputs
     * - @p domain: domain/grid probe providing field access
     * - @p searcher: spatial hashing probe for neighborhood queries
     * - @p particle: particle/fluid probe providing particle data
     *
     * ## Expected behavior
     * Implementations typically:
     * - read relevant simulation quantities,
     * - compute derived statistics or fields,
     * - optionally write results to buffers or diagnostics.
     *
     * @param domain Domain probe (passed by value for device compatibility).
     * @param searcher Spatial hashing probe.
     * @param particle Fluid/particle probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure(DomainDeviceProbe<T> domain,
            SpatialHashingProbe<T> searcher,
            FluidDeviceProbe<T> particle)
        = 0;

    /**
     * @brief Return the measurement mode used by this measure.
     *
     * @details
     * This function informs the caller about the scope of the measurement,
     * allowing higher-level systems to optimize execution or scheduling.
     *
     * @return The current @ref MeasureModeType.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual MeasureModeType
    measure_mode() const noexcept = 0;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::MeasureModeType.
 */
using MeasureModeType = atlas::system::MeasureModeType;

/**
 * @brief Convenience alias for @ref atlas::system::Measure.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Measure = atlas::system::Measure<T>;

/**
 * @brief Host-side shared pointer to a measurement.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MeasureHostPtr = atlas::host_shared_ptr<atlas::system::Measure<T>>;

/**
 * @brief Device-side shared pointer to a measurement.
 *
 * @details
 * Provided for API symmetry; most measurements are orchestrated on the host.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MeasureDevicePtr = atlas::device_shared_ptr<atlas::system::Measure<T>>;

} // namespace atlas