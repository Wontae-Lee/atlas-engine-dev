#pragma once

/**
 * @file sink.h
 * @brief Declares the Sink class used to remove particles from a fluid according to configured sink units.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::fluid {

/**
 * @brief Particle removal sink for a fluid.
 *
 * A Sink manages one or more sink units and removes particles from an
 * associated Fluid instance according to configured despawn rules.
 *
 * Its responsibilities include:
 * - storing sink units and despawn policies,
 * - updating time-dependent sink units,
 * - testing fluid particles against sink geometry,
 * - marking particles as inactive,
 * - triggering particle compaction through the target fluid.
 *
 * @tparam T Floating-point scalar type used by the sink and fluid.
 */
template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    /**
     * @brief Builder for configuring and constructing Sink objects.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Sink() = default;

    /**
     * @brief Destructor.
     */
    ~Sink() = default;

    /**
     * @brief Constructs a sink from fully prepared buffers and configuration.
     *
     * @param units Device buffer containing sink units.
     * @param despawn_types Device buffer containing despawn type configuration.
     * @param despawn_operators Device buffer containing despawn operators.
     * @param fluid Host shared pointer to the target fluid.
     * @param flip Whether despawn acceptance should be inverted.
     * @param tolerance Geometric tolerance used during despawn tests.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sink(DeviceBuffer<Unit<T>> units,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<DespawnOperator<T>> despawn_operators,
         atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
         bool flip   = false,
         T tolerance = T(0)) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent Sink construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Updates all sink units with the given time step.
     *
     * This function is typically used to advance time-dependent unit state
     * before sink processing.
     *
     * @param dt Time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Applies sink logic to the target fluid.
     *
     * This function tests active particles against the configured sink units
     * and despawn operators, marks matching particles as inactive, and then
     * compacts the fluid storage.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink();

private:
    /**
     * @brief Sink units defining sink geometry and transforms.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Despawn type configuration associated with the sink units.
     */
    DeviceBuffer<DespawnType> _despawn_types;

    /**
     * @brief Despawn operators used to decide whether particles should be removed.
     */
    DeviceBuffer<DespawnOperator<T>> _despawn_operators;

    /**
     * @brief Target fluid whose particles are processed by this sink.
     */
    atlas::host_shared_ptr<atlas::Fluid<T>> _fluid;

    /**
     * @brief Whether despawn acceptance should be inverted.
     */
    bool _flip = false;

    /**
     * @brief Geometric tolerance used for despawn tests.
     */
    T _tolerance = T(0);
};

/**
 * @brief Builder for Sink.
 *
 * This builder collects sink units, despawn configuration, target fluid, and
 * sink parameters before constructing a validated Sink object.
 *
 * Validation ensures that:
 * - a target fluid is provided,
 * - at least one sink unit exists,
 * - despawn types are non-empty,
 * - despawn operators are non-empty,
 * - despawn types and operators either have size 1 or match the unit count,
 * - tolerance is finite.
 *
 * @tparam T Floating-point scalar type used by the sink and fluid.
 */
template <typename T>
class Sink<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a validated Sink object.
     *
     * @return Constructed Sink object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sink<T>
    build();

    /**
     * @brief Builds a host-side shared Sink object.
     *
     * @return Host shared pointer to a constructed Sink object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sink<T>>
    make_host_shared();

    /**
     * @brief Appends sink units to the builder.
     *
     * @param units Host buffer containing sink units.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the target fluid.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Appends despawn type configuration entries.
     *
     * Despawn types must either contain exactly one entry shared by all units or
     * one entry per unit.
     *
     * @param despawn_types Host buffer containing despawn type entries.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p despawn_types is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    /**
     * @brief Appends a single despawn operator.
     *
     * @param despawn_operator Despawn operator to append.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept;

    /**
     * @brief Appends multiple despawn operators.
     *
     * Despawn operators must either contain exactly one entry shared by all
     * units or one entry per unit.
     *
     * @param despawn_operators Host buffer containing despawn operators.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p despawn_operators is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators);

    /**
     * @brief Sets the geometric tolerance used during despawn tests.
     *
     * @param tolerance Tolerance used for despawn checks.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Sets whether despawn acceptance should be inverted.
     *
     * @param flip Whether despawn acceptance is flipped.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Host-side sink units collected by the builder.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Target fluid collected by the builder.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Host-side despawn type configuration.
     */
    HostBuffer<DespawnType> _despawn_types;

    /**
     * @brief Host-side despawn operators.
     */
    HostBuffer<DespawnOperator<T>> _despawn_operators;

    /**
     * @brief Whether despawn acceptance should be inverted.
     */
    bool _flip = false;

    /**
     * @brief Tolerance used during geometric despawn checks.
     */
    T _tolerance = T(0);
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::Sink.
 *
 * @tparam T Floating-point scalar type used by the sink.
 */
template <typename T>
using Sink = atlas::fluid::Sink<T>;

/**
 * @brief Host-side shared pointer alias for Sink.
 *
 * @tparam T Floating-point scalar type used by the sink.
 */
template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

/**
 * @brief Device-side shared pointer alias for Sink.
 *
 * @tparam T Floating-point scalar type used by the sink.
 */
template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

} // namespace atlas

#include <atlas/sink/sink.hpp>