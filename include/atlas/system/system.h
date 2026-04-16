#pragma once

#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class System {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(atlas::host_shared_ptr<atlas::Fluid<T>> fluid);

    ATLAS_HOST ATLAS_FORCE_INLINE
    System(atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
           T dt,
           DomainHostPtr<T> domain,
           CodecHostPtr<T> codec,
           SourceHostPtr<T> source,
           SinkHostPtr<T> sink,
           MeasureHostPtr<T> measure,
           ColliderHostPtr<T> collider,
           OrchestratorHostPtr<T> solver);

    ATLAS_HOST ATLAS_FORCE_INLINE ~System() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

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
    set_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid);

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

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<atlas::Fluid<T>>
    fluid() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>&
    particle_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const FluidDeviceProbe<T>&
    particle_probe() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Universe<T>&
    domain_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const Universe<T>&
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
    T _dt { static_cast<T>(0.01) };

    atlas::host_shared_ptr<atlas::Fluid<T>> _fluid {};

    DomainHostPtr<T> _domain {};

    CodecHostPtr<T> _codec {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    FluidDeviceProbe<T> _particle_probe {};

    SpatialHashingProbe<T> _searcher_probe {};

    CodecDeviceProbe<T> _codec_probe {};

    SourceHostPtr<T> _source {};

    SinkHostPtr<T> _sink {};

    MeasureHostPtr<T> _measure {};

    ColliderHostPtr<T> _collider {};

    OrchestratorHostPtr<T> _solver {};
};

template <typename T>
class System<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

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
    T _dt { static_cast<T>(0.01) };

    atlas::host_shared_ptr<atlas::Fluid<T>> _fluid {};

    DomainHostPtr<T> _domain {};

    CodecHostPtr<T> _codec {};

    SourceHostPtr<T> _source {};

    SinkHostPtr<T> _sink {};

    MeasureHostPtr<T> _measure {};

    ColliderHostPtr<T> _collider {};

    OrchestratorHostPtr<T> _solver {};
};

}

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>