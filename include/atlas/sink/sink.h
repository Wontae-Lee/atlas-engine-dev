#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>
#include <atlas/remove/remove.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Runtime particle sink that removes particles based on geometric despawn conditions.
 *
 * @details
 * `Sink<T>` mirrors @ref Source structure:
 * - Bound @ref Unit supplies geometry and transform
 * - @ref DespawnOperator classifies particles (Surface/Volume)
 * - Compacts particles in-place via @ref ParticleDeviceProbe
 *
 * Transforms world positions to unit-local space via `SyncOperator::sync_to_local`,
 * tests against despawn criteria, and removes matching particles.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    class Builder;

public:
    Sink()  = default;
    ~Sink() = default;

    /**
     * @brief Constructs a sink from a unit, despawn type, and tolerance.
     *
     * @param unit Unit defining geometry and transform.
     * @param despawn_type Surface or Volume removal mode.
     * @param tolerance Classification tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sink(Unit<T> unit,
         DespawnType despawn_type = DespawnType::Surface,
         bool flip                = false,
         T tolerance              = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Removes particles matching despawn criteria and compacts the probe.
     *
     * @details
     * Transforms particles to local space, tests against geometry using the despawn
     * operator (Surface/Volume), and removes matching particles via remove_if.
     * Updates `particle_count` in-place.
     *
     * @param particle_probe Target particle storage probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink(ParticleDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(Unit<T> unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_type(DespawnType despawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operator(DespawnOperator<T> despawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    /**
     * @brief Enable or disable inversion of the despawn predicate.
     *
     * @details
     * When `flip` is enabled, particles that would normally be kept are
     * removed, and particles that would normally be removed are kept.
     *
     * @param flip Whether to invert despawn classification.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Unit<T>&
    unit() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DespawnType
    despawn_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DespawnOperator<T>&
    despawn_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    /**
     * @brief Return whether despawn classification is inverted.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

private:
    Unit<T> _unit;
    DespawnOperator<T> _despawn_operator { DespawnType::Surface };
    bool _flip   = false;
    T _tolerance = T(0);
};

template <typename T>
class Sink<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Sink<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sink<T>>
    make_host_shared();

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(const Unit<T>& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(Unit<T>&& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_type(DespawnType despawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<Unit<T>> _unit;
    DespawnType _despawn_type = DespawnType::Surface;
    bool _flip                = false;
    T _tolerance              = T(0);
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using Sink = atlas::system::Sink<T>;

template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

} // namespace atlas

#include <atlas/sink/sink.hpp>
