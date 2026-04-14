#pragma once

/**
 * @file thermometer_operator.h
 * @brief Declares a variance-based thermometer operator and its tagged wrapper for runtime measurement dispatch.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::VarianceThermometerOperator, a concrete measurement
 *   operator that performs variance-oriented thermometer measurements over the
 *   current simulation state,
 * - @ref atlas::system::ThermometerOperator, a tagged-union wrapper used to
 *   store and dispatch thermometer operators through a uniform host-side API.
 *
 * ## Purpose
 * Thermometer operators are used by the Atlas measurement subsystem to inspect
 * runtime particle and domain state and compute temperature-related statistics.
 * In this header, the active concrete implementation is variance-based, meaning
 * that its measurement behavior is centered on a variance-style statistical
 * evaluation defined in the corresponding implementation file.
 *
 * ## Runtime inputs
 * The measurement interface consumes the canonical runtime probes:
 * - @ref DomainDeviceProbe, which exposes domain/grid-aligned field data,
 * - @ref SpatialHashingProbe, which exposes neighborhood lookup information,
 * - @ref FluidDeviceProbe, which exposes particle state.
 *
 * A @ref MeasureModeType selector is also accepted so the measurement can be
 * restricted or configured according to the active measurement policy.
 *
 * ## Tagged-wrapper design
 * Although this header currently stores only one concrete thermometer variant,
 * the @ref ThermometerOperator wrapper preserves the same tagged-union style
 * used elsewhere in Atlas. This offers:
 * - a stable public API,
 * - extensibility for future thermometer variants,
 * - explicit control over object lifetime for union members.
 *
 * ## Host-only interface
 * All entry points are marked `ATLAS_HOST`, indicating that construction,
 * dispatch, and measurement orchestration are intended for host-side execution.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout the measurement pipeline.
 */

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/thermometer/thermometer_type.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

/**
 * @brief Concrete thermometer operator that performs variance-based measurement.
 *
 * @details
 * @ref VarianceThermometerOperator is a host-side measurement operator used by
 * the Atlas measurement subsystem to evaluate variance-oriented temperature or
 * temperature-like statistics from the active simulation state.
 *
 * The exact measurement algorithm is implementation-defined in the corresponding
 * `.hpp` file, but it conceptually operates on:
 * - domain/grid information from @ref DomainDeviceProbe,
 * - neighbor-search information from @ref SpatialHashingProbe,
 * - particle data from @ref FluidDeviceProbe.
 *
 * ## Intended role
 * This operator is suitable for measurement paths where the quantity of interest
 * depends on statistical spread or fluctuation rather than only a simple mean.
 *
 * ## Measurement mode
 * The @p measure_mode parameter allows callers to specify which subset or style
 * of measurement should be applied. The meaning of individual
 * @ref MeasureModeType values is defined elsewhere in the Atlas measurement
 * subsystem.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation state and measurement logic.
 */
template <typename T>
struct VarianceThermometerOperator final {
    /**
     * @brief Perform a variance-based thermometer measurement.
     *
     * @details
     * Consumes the current runtime probes and applies the variance thermometer's
     * measurement logic according to the specified @p measure_mode.
     *
     * The exact side effects and result storage policy are implementation-defined.
     * Typical behaviors may include:
     * - accumulating measurement results into domain fields,
     * - writing derived statistics to measurement buffers,
     * - updating runtime diagnostic state.
     *
     * @param domain Domain probe providing grid metadata and field access.
     * @param searcher Spatial hashing probe providing neighborhood lookup support.
     * @param particle Fluid probe providing particle positions, velocities,
     *        temperatures, species, and active counts.
     * @param measure_mode Measurement mode selector controlling how the
     *        variance-based measurement should be applied. Defaults to
     *        @ref MeasureModeType::All.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(const DomainDeviceProbe<T>& domain,
            const SpatialHashingProbe<T>& searcher,
            const FluidDeviceProbe<T>& particle,
            MeasureModeType measure_mode = MeasureModeType::All) const;
};

/**
 * @brief Tagged-union wrapper for thermometer measurement operators.
 *
 * @details
 * @ref ThermometerOperator stores one active thermometer operator together with
 * a runtime @ref ThermometerType tag describing which concrete implementation is
 * currently active.
 *
 * In this specialization of the public interface, only the variance-based
 * thermometer is stored. Even so, the wrapper preserves the tagged-union pattern
 * so that:
 * - the API remains structurally consistent with other Atlas operator wrappers,
 * - future thermometer variants can be added without changing the outer usage
 *   pattern,
 * - callers interact through one uniform @ref measure entry point.
 *
 * ## Active state
 * The active thermometer is selected by @ref type. In this header, the only
 * stored concrete member is:
 * - @ref variance
 *
 * ## Responsibilities
 * The wrapper is responsible for:
 * - constructing the active thermometer from a default state or runtime tag,
 * - copying the active thermometer correctly,
 * - destroying the active thermometer correctly,
 * - dispatching measurement calls to the active implementation.
 *
 * ## Typical usage
 * Higher-level code can store a `ThermometerOperator<T>` and invoke:
 * @code
 * thermometer.measure(domain_probe, searcher_probe, particle_probe, MeasureModeType::All);
 * @endcode
 * without needing to directly reference the concrete thermometer type.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the underlying thermometer implementation.
 */
template <typename T>
struct ThermometerOperator final {
    /**
     * @brief Runtime tag identifying the active thermometer implementation.
     *
     * @details
     * The default value selects the variance thermometer.
     */
    ThermometerType type = ThermometerType::Variance;

    /**
     * @brief Union storage for the active thermometer implementation.
     *
     * @details
     * In this header, only the variance thermometer is stored as a concrete
     * union member.
     */
    union {
        /**
         * @brief Variance-based thermometer implementation.
         */
        VarianceThermometerOperator<T> variance;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a thermometer operator with the default active type,
     * @ref ThermometerType::Variance, and initializes the corresponding union
     * member.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    ThermometerOperator() noexcept;

    /**
     * @brief Construct a thermometer operator from a runtime thermometer type tag.
     *
     * @details
     * Initializes the active union member corresponding to @p type_.
     *
     * Since this header currently only stores the variance thermometer, supported
     * behavior for other tag values is implementation-defined.
     *
     * @param type_ Runtime thermometer type to activate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit ThermometerOperator(ThermometerType type_) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies the active thermometer state from @p other, preserving both the
     * runtime tag and the active union member.
     *
     * @param other Source thermometer operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    ThermometerOperator(const ThermometerOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Replaces the current active thermometer state with a copy of the state
     * stored in @p other.
     *
     * Any currently active union member is destroyed before the new state is
     * copied in.
     *
     * @param other Source thermometer operator.
     * @return Reference to `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ThermometerOperator&
    operator=(const ThermometerOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * @details
     * Destroys the currently active union member according to the runtime tag
     * stored in @ref type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ~ThermometerOperator() noexcept;

    /**
     * @brief Construct a thermometer operator from a concrete variance thermometer.
     *
     * @details
     * Activates the @ref variance union member and sets the runtime type
     * accordingly.
     *
     * @param op Variance thermometer operator to store.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit ThermometerOperator(const VarianceThermometerOperator<T>& op) noexcept;

    /**
     * @brief Dispatch a measurement call to the active thermometer implementation.
     *
     * @details
     * Forwards the supplied runtime probes and measurement mode to the currently
     * active thermometer implementation.
     *
     * In this header, that dispatch targets @ref variance.
     *
     * @param domain Domain probe providing field and grid access.
     * @param searcher Spatial hashing probe providing neighborhood lookup data.
     * @param particle Fluid probe providing particle state.
     * @param measure_mode Measurement mode selector controlling the behavior of
     *        the active thermometer. Defaults to @ref MeasureModeType::All.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(const DomainDeviceProbe<T>& domain,
            const SpatialHashingProbe<T>& searcher,
            const FluidDeviceProbe<T>& particle,
            MeasureModeType measure_mode = MeasureModeType::All) const;

private:
    /**
     * @brief Destroy the currently active union member.
     *
     * @details
     * Uses the runtime type tag stored in @ref type to determine which union
     * member is active, then invokes the matching destructor.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Copy the active thermometer state from another operator.
     *
     * @details
     * Reconstructs the correct union member in this object according to the
     * runtime type tag stored in @p other.
     *
     * @param other Source thermometer operator whose active state is copied.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    copy_from(const ThermometerOperator& other) noexcept;
};

} // namespace atlas::system

#include <atlas/measure/thermometer/thermometer_operator.hpp>