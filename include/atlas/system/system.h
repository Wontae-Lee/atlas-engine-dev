#pragma once

/**
 * @file system.h
 * @brief Declares the high-level simulation runtime orchestration object.
 *
 * System ties together fluid storage, domain data, search structures, codecs,
 * emitters, sinks, measurement, colliders, and solvers. It owns the canonical
 * device probes for the active simulation and executes the staged update loop
 * used by Atlas runtime examples and applications.
 */

#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief High-level runtime object that orchestrates a simulation step.
 *
 * System owns or references the major simulation subsystems and runs them in a
 * fixed order through update(). It also caches the single authoritative device
 * probes for the fluid, domain, searcher, and codec so backend code can be
 * dispatched without repeatedly rebuilding runtime views.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class System {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    class Builder;

public:
    /**
     * @brief Constructs a system with only fluid storage.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(FluidHostPtr<T> fluid);

    /**
     * @brief Constructs a fully configured system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    System(FluidHostPtr<T> fluid,
           T dt,
           DomainHostPtr<T> domain,
           CodecHostPtr<T> codec,
           SourceHostPtr<T> source,
           SinkHostPtr<T> sink,
           MeasureHostPtr<T> measure,
           ColliderHostPtr<T> collider,
           OrchestratorHostPtr<T> solver);

    /// @brief Destructor.
    ATLAS_HOST ATLAS_FORCE_INLINE ~System() = default;

    /// @brief Returns a builder initialized with default values.
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Executes one full simulation step.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    search();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_dt(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_domain(const DomainHostPtr<T>& domain);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(const CodecHostPtr<T>& codec);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_source(const SourceHostPtr<T>& source);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sink(const SinkHostPtr<T>& sink);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measure(const MeasureHostPtr<T>& measure);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_solver(const OrchestratorHostPtr<T>& solver);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidHostPtr<T>
    fluid() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>&
    particle_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const FluidDeviceProbe<T>&
    particle_probe() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>&
    domain_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const DomainDeviceProbe<T>&
    domain_probe() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingProbe<T>&
    searcher_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const SpatialHashingProbe<T>&
    searcher_probe() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>&
    codec_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const CodecDeviceProbe<T>&
    codec_probe() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    dt() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DomainHostPtr<T>&
    domain() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SourceHostPtr<T>&
    source() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SinkHostPtr<T>&
    sink() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MeasureHostPtr<T>&
    measure() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const ColliderHostPtr<T>&
    collider() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const OrchestratorHostPtr<T>&
    solver() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_source() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_sink() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_measure() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_collider() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_solver() noexcept;

private:
    T _dt { static_cast<T>(0.01) }; ///< Simulation timestep.

    FluidHostPtr<T> _fluid {}; ///< Owned fluid subsystem.
    DomainHostPtr<T> _domain {}; ///< Owned domain subsystem.
    CodecHostPtr<T> _codec {}; ///< Owned codec subsystem.
    SpatialHashingSearcherHostPtr<T> _searcher {}; ///< Derived spatial search structure.

    FluidDeviceProbe<T> _particle_probe {}; ///< Canonical fluid device probe.
    DomainDeviceProbe<T> _domain_probe {}; ///< Canonical domain device probe.
    SpatialHashingProbe<T> _searcher_probe {}; ///< Canonical searcher device probe.
    CodecDeviceProbe<T> _codec_probe {}; ///< Canonical codec device probe.

    SourceHostPtr<T> _source {}; ///< Optional source stage.
    SinkHostPtr<T> _sink {}; ///< Optional sink stage.
    MeasureHostPtr<T> _measure {}; ///< Optional measurement stage.
    ColliderHostPtr<T> _collider {}; ///< Optional collision stage.
    OrchestratorHostPtr<T> _solver {}; ///< Optional solver stage.
};

template <typename T>
class System<T>::Builder final {
public:
    /// @brief Default constructor.
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const DomainHostPtr<T>& domain);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(const CodecHostPtr<T>& codec);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure(const MeasureHostPtr<T>& measure);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(const OrchestratorHostPtr<T>& solver);

    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _dt { static_cast<T>(0.01) }; ///< Pending timestep.

    FluidHostPtr<T> _fluid {}; ///< Pending fluid subsystem.
    DomainHostPtr<T> _domain {}; ///< Pending domain subsystem.
    CodecHostPtr<T> _codec {}; ///< Pending codec subsystem.

    SourceHostPtr<T> _source {}; ///< Pending source subsystem.
    SinkHostPtr<T> _sink {}; ///< Pending sink subsystem.
    MeasureHostPtr<T> _measure {}; ///< Pending measure subsystem.
    ColliderHostPtr<T> _collider {}; ///< Pending collider subsystem.
    OrchestratorHostPtr<T> _solver {}; ///< Pending solver subsystem.
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::System.
 */
template <typename T>
using System = system::System<T>;

/**
 * @brief Host shared pointer alias for System.
 */
template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>
