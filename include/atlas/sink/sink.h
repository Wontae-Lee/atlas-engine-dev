#pragma once

/**
 * @file sink.h
 * @brief Declares the Sink class used to remove particles from a fluid according
 *        to configured sink units and despawn policies.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <type_traits>

namespace atlas::fluid {

/**
 * @brief Removes particles from a target fluid according to configured sink units.
 *
 * A Sink combines:
 * - geometric sink units,
 * - despawn operators describing the removal rule,
 * - a target fluid whose particles are tested and possibly removed.
 *
 * Its responsibilities include:
 * - advancing time-dependent sink units,
 * - testing active particles against sink geometry,
 * - deciding whether each particle should survive or be removed,
 * - updating the active-state mask,
 * - compacting all registered fluid states so surviving particles occupy a dense prefix.
 *
 * Sink supports two common despawn configuration modes:
 * - one shared despawn rule for all sink units,
 * - one despawn rule per sink unit.
 *
 * It also supports a @ref flip mode that inverts the survival logic:
 * - normal mode: matching particles are removed,
 * - flipped mode: matching particles are kept and non-matching particles are removed.
 *
 * @tparam T Floating-point scalar type used by the sink and fluid.
 */
template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    /**
     * @brief Cached raw views over common sink runtime data.
     *
     * Sink processing repeatedly needs the same unit, despawn-operator, and
     * particle state buffers. This probe groups those values so device lambdas
     * can capture one compact object by value.
     */
    struct SinkProbe {
        const Unit<T>* units {};
        const DespawnOperator<T>* despawn_operators {};
        const Vector3<T>* positions {};
        const Vector3<T>* velocities {};
        int* active {};

        int unit_count {};
        int despawn_operator_count {};
        std::size_t particle_count {};

        bool flip {};
        T tolerance {};
        T time_step {};
    };

    /**
     * @brief Builder for validated Sink construction.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty sink with no units, no despawn configuration,
     * and no target fluid.
     */
    Sink() = default;

    /**
     * @brief Destructor.
     */
    ~Sink() = default;

    /**
     * @brief Constructs a sink from prepared device buffers and runtime parameters.
     *
     * @param units Device buffer containing sink units.
     * @param despawn_types Device buffer containing despawn type tags.
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
         bool flip                = false,
         T tolerance              = T(0),
         ObserverHostPtr observer = nullptr) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Newly created builder object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Advances sink units and then applies sink processing for the given step.
     *
     * The default update flow is:
     * 1. update all sink units using Unit::update(dt),
     * 2. test fluid particles against the sink,
     * 3. compact fluid storage after marking removed particles inactive.
     *
     * @param dt Positive simulation time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Applies sink logic to the target fluid.
     *
     * This function:
     * - reads particle positions and active flags,
     * - evaluates each active particle against all sink units,
     * - writes updated active flags,
     * - compacts the fluid so surviving particles occupy a dense prefix.
     *
     * Removal is controlled by:
     * - the configured despawn operators,
     * - the sink-unit transforms and geometry,
     * - the optional flip flag,
     * - the configured geometric tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink(T dt = T(0));

    /**
     * @brief Compacts the target fluid so surviving particles occupy a dense prefix.
     *
     * This compaction step is driven by the fluid's active-state mask.
     * The method performs the following steps:
     * 1. convert active flags into a binary keep mask,
     * 2. exclusive-scan the keep mask into destination offsets,
     * 3. build a compacted destination-to-source index map,
     * 4. compact every registered fluid state using the same index map,
     * 5. clear the inactive tail and update the logical particle count.
     *
     * This helper is public only because of NVCC limitations around certain
     * lambda/device contexts in private member functions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compact_fluid_particles();

    /**
     * @brief Build a cached probe for sink unit and particle-state data.
     *
     * @param probe Output probe populated with raw pointers and scalar metadata.
     * @return True when all required sink and fluid state exists.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe(SinkProbe& probe, T dt = T(0)) noexcept;

private:
    /**
     * @brief Sink units defining geometry and transforms used for despawn tests.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Runtime despawn-type tags associated with the sink configuration.
     *
     * This buffer is kept as part of the sink configuration surface even though
     * current despawn dispatch primarily uses @ref _despawn_operators.
     */
    DeviceBuffer<DespawnType> _despawn_types;

    /**
     * @brief Despawn operators used to decide whether particles should be removed.
     *
     * This may contain:
     * - exactly one shared operator for all units,
     * - or one operator per unit.
     */
    DeviceBuffer<DespawnOperator<T>> _despawn_operators;

    /**
     * @brief Target fluid whose particles are processed by this sink.
     */
    atlas::host_shared_ptr<atlas::Fluid<T>> _fluid;

    /**
     * @brief Optional observer used to record sink metrics.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Whether despawn acceptance should be inverted.
     *
     * When false:
     * - despawn match => particle removed
     * - no match      => particle kept
     *
     * When true:
     * - despawn match => particle kept
     * - no match      => particle removed
     */
    bool _flip = false;

    /**
     * @brief Geometric tolerance passed to despawn tests.
     */
    T _tolerance = T(0);

    /**
     * @brief Scratch keep-mask buffer used during sink-driven compaction.
     *
     * Each entry is:
     * - 1 if the particle survives,
     * - 0 if the particle is removed.
     */
    DeviceBuffer<std::size_t> _keep;

    /**
     * @brief Scratch prefix-offset buffer used during sink-driven compaction.
     *
     * This buffer stores the exclusive scan of @ref _keep and provides the
     * destination index of each surviving particle in the compacted layout.
     */
    DeviceBuffer<std::size_t> _offsets;

    /**
     * @brief Scratch destination-to-source index map used during sink-driven compaction.
     *
     * After construction, entry k stores the original particle index that should
     * be moved into compacted destination slot k.
     */
    DeviceBuffer<std::size_t> _compact_indices;

    /**
     * @brief Scratch buffer storing the matched sink-unit index for each particle.
     */
    DeviceBuffer<int> _despawned_unit_indices;

    /**
     * @brief Monotonic sink-step index used by sink metric recording.
     */
    std::size_t _step_index = 0;
};

/**
 * @brief Builder for validated Sink construction.
 *
 * The builder collects:
 * - sink units,
 * - the target fluid,
 * - despawn types,
 * - despawn operators,
 * - sink tolerance,
 * - optional flip behavior.
 *
 * Validation ensures that:
 * - a target fluid is provided,
 * - at least one sink unit is present,
 * - despawn types are non-empty,
 * - despawn operators are non-empty,
 * - despawn types either have size 1 or match the unit count,
 * - despawn operators either have size 1 or match the unit count,
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
     * @brief Validates the current builder configuration and constructs a Sink object.
     *
     * @return Constructed Sink object.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sink<T>
    build();

    /**
     * @brief Builds a Sink object and wraps it in host-managed shared storage.
     *
     * @return Host shared pointer to the constructed Sink object.
     *
     * @throw std::runtime_error Thrown if validation fails.
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
     * @brief Sets the target fluid processed by the sink.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Sets the optional observer used to record sink metrics.
     *
     * @param observer Host shared pointer to the observer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /**
     * @brief Appends despawn-type configuration entries.
     *
     * Despawn types must contain either:
     * - one shared entry for all units,
     * - or one entry per unit.
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
     * Despawn operators must contain either:
     * - one shared entry for all units,
     * - or one entry per unit.
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
     * @param tolerance Tolerance passed to despawn queries.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Sets whether despawn acceptance should be inverted.
     *
     * @param flip Whether the normal despawn/keep interpretation is inverted.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if any required invariant is violated.
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
     * @brief Optional observer collected by the builder.
     */
    ObserverHostPtr _observer {};

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
     * @brief Tolerance used by geometric despawn tests.
     */
    T _tolerance = T(0);
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Sink = atlas::fluid::Sink<T>;

/**
 * @brief Host-side shared pointer alias for Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

/**
 * @brief Device-side shared pointer alias for Sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

} // namespace atlas

#include <atlas/sink/sink.hpp>
