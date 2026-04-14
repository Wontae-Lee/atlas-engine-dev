#pragma once

/**
 * @file thermometer.h
 * @brief Declares the abstract base interface for thermometer-based measurement components.
 *
 * @details
 * This header defines @ref atlas::system::Thermometer, an abstract specialization
 * of the generic @ref atlas::system::Measure interface used to perform
 * temperature-related or temperature-like statistical measurements within the
 * Atlas simulation framework.
 *
 * ## Conceptual role
 * A thermometer represents a measurement object that:
 * - consumes runtime simulation probes (domain, spatial structure, particles),
 * - evaluates a temperature-related metric,
 * - optionally writes results into domain fields, buffers, or diagnostic state.
 *
 * Unlike low-level operator structs, this class:
 * - provides a polymorphic interface,
 * - can be stored via shared pointers,
 * - integrates into higher-level measurement orchestration pipelines.
 *
 * ## Relationship to Measure
 * `Thermometer<T>` inherits from @ref Measure<T> and therefore participates in
 * the generic measurement system used by Atlas.
 *
 * It refines the interface by:
 * - enforcing a temperature-specific semantic contract,
 * - introducing @ref ThermometerType for runtime identification,
 * - exposing @ref is_valid for state validation.
 *
 * ## Measurement pipeline
 * All thermometer implementations operate on the same runtime probe set:
 * - @ref DomainDeviceProbe for grid/field data,
 * - @ref SpatialHashingProbe for neighborhood queries,
 * - @ref FluidDeviceProbe for particle state.
 *
 * These probes are passed to @ref measure, which derived classes must implement.
 *
 * ## Measurement mode
 * The class stores a @ref MeasureModeType that determines how measurement is
 * performed (e.g., full-domain vs. partial measurement, or other policy-based
 * variations defined elsewhere).
 *
 * ## Design goals
 * - Provide a stable polymorphic interface for measurement components,
 * - Enable runtime selection and composition of measurement strategies,
 * - Separate high-level orchestration from low-level measurement operators.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the measurement system.
 */

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/measure/thermometer/thermometer_type.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Abstract base class for thermometer measurement components.
 *
 * @details
 * `Thermometer<T>` defines the interface for all temperature-related measurement
 * objects in Atlas. It extends @ref Measure<T> by specializing the semantics
 * toward thermometer-style evaluation.
 *
 * ## Responsibilities
 * Derived classes must:
 * - implement @ref measure to perform the actual measurement,
 * - implement @ref is_valid to report whether the thermometer is usable,
 * - implement @ref type to report their runtime thermometer type.
 *
 * ## Measurement contract
 * The @ref measure function is expected to:
 * - inspect the provided domain, searcher, and particle probes,
 * - compute temperature-related quantities (e.g., average, variance, etc.),
 * - update measurement outputs (implementation-defined).
 *
 * ## Measurement mode handling
 * The base class stores the current @ref MeasureModeType in @ref _measure_mode.
 * This value:
 * - can be set via @ref set_measure_mode,
 * - is returned through @ref measure_mode,
 * - is typically used by derived implementations to control behavior.
 *
 * ## Polymorphic usage
 * Thermometer objects are intended to be used via:
 * - value semantics (for lightweight configurations), or
 * - shared pointers (for runtime polymorphism and ownership).
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation and measurement system.
 */
template <typename T>
class Thermometer : public Measure<T> {
    static_assert(std::is_floating_point_v<T>, "Thermometer requires a floating-point T");

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the thermometer with the default measurement mode
     * @ref MeasureModeType::All.
     */
    Thermometer() = default;

    /**
     * @brief Construct a thermometer with a specific measurement mode.
     *
     * @param measure_mode Measurement mode to use for subsequent calls to @ref measure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Thermometer(MeasureModeType measure_mode) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Thermometer() override = default;

    /**
     * @brief Perform a thermometer measurement.
     *
     * @details
     * This pure virtual function must be implemented by derived classes to
     * evaluate temperature-related quantities based on the current simulation state.
     *
     * ## Inputs
     * - @p domain: domain/grid probe providing field access,
     * - @p searcher: spatial hashing probe for neighborhood queries,
     * - @p particle: particle/fluid probe providing particle state.
     *
     * ## Expected behavior
     * Implementations typically:
     * - read particle velocities, energies, or other relevant quantities,
     * - compute derived temperature statistics,
     * - store or accumulate results in domain or measurement buffers.
     *
     * @param domain Domain probe (passed by value for device compatibility).
     * @param searcher Spatial hashing probe (passed by value).
     * @param particle Fluid/particle probe (passed by value).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(DomainDeviceProbe<T> domain,
            SpatialHashingProbe<T> searcher,
            FluidDeviceProbe<T> particle) override = 0;

    /**
     * @brief Check whether the thermometer is in a valid state.
     *
     * @details
     * Derived classes must define what constitutes a valid thermometer.
     * Typical checks may include:
     * - presence of required operators or parameters,
     * - consistency of internal configuration,
     * - availability of required runtime dependencies.
     *
     * @return `true` if the thermometer is valid and can perform measurement,
     *         otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept = 0;

    /**
     * @brief Return the runtime thermometer type.
     *
     * @details
     * This function allows callers to identify the specific thermometer
     * implementation at runtime without relying on RTTI.
     *
     * @return A value of @ref ThermometerType identifying the concrete implementation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual ThermometerType
    type() const noexcept = 0;

    /**
     * @brief Set the measurement mode.
     *
     * @details
     * Updates the internal @ref _measure_mode, which controls how measurement
     * is performed by derived implementations.
     *
     * @param measure_mode New measurement mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measure_mode(MeasureModeType measure_mode) noexcept;

    /**
     * @brief Return the current measurement mode.
     *
     * @details
     * Overrides @ref Measure::measure_mode to expose the stored measurement mode.
     *
     * @return Current measurement mode.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

protected:
    /**
     * @brief Stored measurement mode.
     *
     * @details
     * Controls how derived thermometer implementations perform measurement.
     * Defaults to @ref MeasureModeType::All.
     */
    MeasureModeType _measure_mode { MeasureModeType::All };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Thermometer.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using Thermometer = atlas::system::Thermometer<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to a thermometer.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using ThermometerHostPtr = atlas::host_shared_ptr<atlas::system::Thermometer<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to a thermometer.
 *
 * @details
 * Provided for API symmetry; thermometer orchestration is typically host-driven.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using ThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::Thermometer<T>>;

} // namespace atlas

#include <atlas/measure/thermometer.hpp>