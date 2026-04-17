#pragma once

#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class System final {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    class Builder;

public:
    System() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    System(FluidHostPtr<T> fluid,
           UniverseHostPtr<T> universe,
           SourceHostPtr<T> source,
           SinkHostPtr<T> sink,
           ColliderHostPtr<T> collider,
           OrchestratorHostPtr<T> orchestrator,
           T dt = static_cast<T>(0.01));

    ~System() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    advect();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    dt() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

private:
    FluidHostPtr<T> _fluid {};

    UniverseHostPtr<T> _universe {};

    SourceHostPtr<T> _source {};

    SinkHostPtr<T> _sink {};

    ColliderHostPtr<T> _collider {};

    OrchestratorHostPtr<T> _orchestrator {};

    T _dt { static_cast<T>(0.01) };
};

template <typename T>
class System<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(const FluidHostPtr<T>& fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const UniverseHostPtr<T>& universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(const OrchestratorHostPtr<T>& orchestrator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    FluidHostPtr<T> _fluid {};

    UniverseHostPtr<T> _universe {};

    SourceHostPtr<T> _source {};

    SinkHostPtr<T> _sink {};

    ColliderHostPtr<T> _collider {};

    OrchestratorHostPtr<T> _orchestrator {};

    T _dt { static_cast<T>(0.01) };
};

}

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>
