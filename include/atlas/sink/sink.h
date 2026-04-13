#pragma once

/**
 * @file sink.h
 * @brief Declares particle removal sinks and builder utilities for despawning particles from a fluid buffer.
 *
 * @details
 * This header defines @ref atlas::system::Sink, a runtime object responsible
 * for removing particles that satisfy configured despawn criteria.
 *
 * A sink combines:
 * - one or more scene units providing local geometry and world pose,
 * - per-unit despawn type configuration,
 * - per-unit despawn operators,
 * - source-wide inversion and tolerance controls.
 *
 * ## High-level responsibilities
 * A sink can:
 * - update the motion state of its configured units,
 * - evaluate whether active particles satisfy one or more despawn conditions,
 * - apply those tests in the local space of each unit,
 * - compact the active particle prefix in-place using remove-if style semantics.
 *
 * ## Particle-removal model
 * The sink operates on the active particle prefix contained in a
 * @ref FluidDeviceProbe. Particles classified for removal are excluded from the
 * surviving prefix, and the remaining active particles are compacted toward the
 * front of the particle buffers.
 *
 * ## Local-space evaluation
 * Despawn tests are performed relative to unit-local geometry. A unit's sync
 * transform defines how world-space particle positions are interpreted with
 * respect to the unit's local coordinate frame.
 *
 * ## Flip mode
 * The @ref flip flag allows despawn decisions to be inverted. This is useful
 * when a sink should remove particles from the complement of a region rather
 * than from the region itself.
 *
 * ## Tolerance
 * The configured tolerance is passed to despawn logic to make geometric tests
 * more robust near boundaries.
 *
 * ## Construction
 * A sink may be:
 * - default-constructed,
 * - constructed directly from device-resident unit and despawn buffers,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Removes particles that satisfy configured despawn conditions.
 *
 * @details
 * @ref Sink is the particle-removal stage used to erase particles matching one
 * or more geometric despawn rules.
 *
 * The sink stores:
 * - the units that define local despawn geometry and pose,
 * - the despawn type for each unit,
 * - the despawn operator for each unit,
 * - global sink configuration such as flip mode and geometric tolerance.
 *
 * ## Typical workflow
 * A call to @ref sink conceptually performs the following steps:
 * 1. transform or interpret particle positions with respect to each unit,
 * 2. evaluate the active despawn rule(s),
 * 3. combine those rule results according to the implementation policy,
 * 4. remove particles that satisfy the selected condition,
 * 5. compact the active particle prefix in-place.
 *
 * ## Update semantics
 * The @ref update function may be used to advance unit motion over time before
 * the despawn pass is applied.
 *
 * ## Empty sinks
 * A sink with no units or no despawn configuration is typically considered empty
 * and may act as a no-op.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Sink.
     *
     * @details
     * The builder stages units, despawn types, despawn operators, and sink-wide
     * parameters, validates them, and constructs either:
     * - a sink by value, or
     * - a host-owned shared pointer to a sink.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty sink with default flip mode and zero tolerance.
     */
    Sink() = default;

    /**
     * @brief Destructor.
     */
    ~Sink() = default;

    /**
     * @brief Construct a sink from explicit units and despawn configuration.
     *
     * @param units Device-resident despawn units.
     * @param despawn_types Despawn type per unit.
     * @param despawn_operators Despawn operator per unit.
     * @param flip Whether despawn decisions are inverted.
     * @param tolerance Geometric tolerance used during despawn tests.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sink(DeviceBuffer<Unit<T>> units,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<DespawnOperator<T>> despawn_operators,
         bool flip   = false,
         T tolerance = T(0)) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Update sink state by one timestep.
     *
     * @details
     * Typically advances unit motion or other time-dependent sink state according
     * to the implementation in `sink.hpp`.
     *
     * @param dt Simulation timestep.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Apply particle removal to a fluid device probe.
     *
     * @details
     * Evaluates despawn rules for active particles and compacts the surviving
     * particle prefix in-place.
     *
     * @param particle_probe Mutable fluid device probe to update in-place.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink(FluidDeviceProbe<T>& particle_probe);

    /**
     * @brief Replace the configured units with a device buffer.
     *
     * @param units New device-resident unit buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    /**
     * @brief Replace the configured units with host-provided values.
     *
     * @param units New host-side unit buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Replace the despawn operators with a device buffer.
     *
     * @param despawn_operators New device-resident despawn-operator buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operators(DeviceBuffer<DespawnOperator<T>> despawn_operators) noexcept;

    /**
     * @brief Replace the despawn operators with host-provided values.
     *
     * @param despawn_operators New host-side despawn-operator buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators);

    /**
     * @brief Replace the despawn types with a device buffer.
     *
     * @param despawn_types New device-resident despawn-type buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_types(DeviceBuffer<DespawnType> despawn_types) noexcept;

    /**
     * @brief Replace the despawn types with host-provided values.
     *
     * @param despawn_types New host-side despawn-type buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    /**
     * @brief Set one despawn operator to be used uniformly.
     *
     * @details
     * The implementation may replicate this operator across all configured units.
     *
     * @param despawn_operator Despawn operator to apply.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept;

    /**
     * @brief Set the geometric despawn tolerance.
     *
     * @param tolerance New sink tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    /**
     * @brief Set whether despawn decisions should be inverted.
     *
     * @param flip New flip mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    /**
     * @brief Return mutable access to the configured units.
     *
     * @return Mutable device buffer of units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    /**
     * @brief Return const access to the configured units.
     *
     * @return Const device buffer of units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    /**
     * @brief Return mutable access to the despawn operators.
     *
     * @return Mutable device buffer of despawn operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<DespawnOperator<T>>&
    despawn_operators() noexcept;

    /**
     * @brief Return const access to the despawn operators.
     *
     * @return Const device buffer of despawn operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<DespawnOperator<T>>&
    despawn_operators() const noexcept;

    /**
     * @brief Return mutable access to the despawn types.
     *
     * @return Mutable device buffer of despawn types.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<DespawnType>&
    despawn_types() noexcept;

    /**
     * @brief Return const access to the despawn types.
     *
     * @return Const device buffer of despawn types.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<DespawnType>&
    despawn_types() const noexcept;

    /**
     * @brief Return the configured geometric tolerance.
     *
     * @return Sink tolerance.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    /**
     * @brief Return whether despawn decisions are inverted.
     *
     * @return `true` if flip mode is enabled; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

    /**
     * @brief Return whether the sink has usable despawn configuration.
     *
     * @return `true` if the sink is empty or inactive; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    /**
     * @brief Despawn units.
     *
     * @details
     * Each unit contributes local geometry and pose for despawn classification.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Despawn type configuration.
     *
     * @details
     * Stores the active despawn classification mode per unit.
     */
    DeviceBuffer<DespawnType> _despawn_types;

    /**
     * @brief Despawn rules.
     *
     * @details
     * Stores the concrete despawn operators evaluated against particles.
     */
    DeviceBuffer<DespawnOperator<T>> _despawn_operators;

    /**
     * @brief Whether despawn decisions are inverted.
     */
    bool _flip = false;

    /**
     * @brief Geometric tolerance used during despawn tests.
     */
    T _tolerance = T(0);
};

/**
 * @brief Fluent builder for @ref Sink.
 *
 * @details
 * The builder provides a controlled construction path for particle sinks by
 * staging:
 * - despawn units,
 * - despawn types,
 * - despawn operators,
 * - sink-wide inversion and tolerance parameters.
 *
 * ## Typical usage
 * @code
 * auto sink = atlas::Sink<float>::builder()
 *     .with_units(units)
 *     .with_despawn_types(types)
 *     .with_despawn_operators(ops)
 *     .with_tolerance(0.0f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the unit list is not empty,
 * - despawn-type and despawn-operator counts are compatible with the unit count,
 * - tolerance is physically admissible.
 *
 * The exact validation rules are implementation-defined in `sink.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Sink<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with empty unit/despawn buffers and default sink
     * parameters.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Sink by value after validation.
     *
     * @return Constructed sink value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sink<T>
    build();

    /**
     * @brief Build a configured @ref Sink in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Sink<T>>` owning the constructed sink.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sink<T>>
    make_host_shared();

    /**
     * @brief Set the despawn units from host-side values.
     *
     * @param units Host-side unit buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Set the per-unit despawn types.
     *
     * @param despawn_types Host-side despawn-type buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    /**
     * @brief Set one despawn operator to be applied uniformly.
     *
     * @details
     * The builder may replicate this operator as needed for all configured units.
     *
     * @param despawn_operator Despawn operator to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept;

    /**
     * @brief Set the per-unit despawn operators.
     *
     * @param despawn_operators Host-side despawn-operator buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators);

    /**
     * @brief Set the geometric despawn tolerance.
     *
     * @param tolerance Tolerance to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Set whether despawn decisions should be inverted.
     *
     * @param flip Flip mode to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on units, despawn metadata, and sink-wide
     * configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending host-side units.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Pending host-side despawn types.
     */
    HostBuffer<DespawnType> _despawn_types;

    /**
     * @brief Pending host-side despawn operators.
     */
    HostBuffer<DespawnOperator<T>> _despawn_operators;

    /**
     * @brief Pending flip mode.
     */
    bool _flip = false;

    /**
     * @brief Pending geometric tolerance.
     */
    T _tolerance = T(0);
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Sink = atlas::system::Sink<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

} // namespace atlas

#include <atlas/sink/sink.hpp>