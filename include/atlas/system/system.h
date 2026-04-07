#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/codec/single_codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/data/particle_data.h>
#include <atlas/domain/domain.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class System {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    class Builder;

public:
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(size_t buffer_size);

    ATLAS_HOST ATLAS_FORCE_INLINE
    System(size_t buffer_size,
           T dt,
           DomainHostPtr<T> domain,
           CodecHostPtr<T> codec,
           HostBuffer<SourceHostPtr<T>> sources,
           HostBuffer<SinkHostPtr<T>> sinks,
           HostBuffer<ColliderHostPtr<T>> colliders) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ~System() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    advect() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_dt(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_domain(const DomainHostPtr<T>& domain);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(const CodecHostPtr<T>& codec);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_source(const SourceHostPtr<T>& source);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sources(const HostBuffer<SourceHostPtr<T>>& sources);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_sink(const SinkHostPtr<T>& sink);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sinks(const HostBuffer<SinkHostPtr<T>>& sinks);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_colliders(const HostBuffer<Collider<T>>& colliders);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE ParticleDataHostPtr<T>
    particle_data() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>&
    particle_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const ParticleDeviceProbe<T>&
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

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SourceHostPtr<T>>&
    sources() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SinkHostPtr<T>>&
    sinks() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<ColliderHostPtr<T>>&
    colliders() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_sources() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_sinks() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_colliders() noexcept;

private:
    T _dt { static_cast<T>(0.01) };

    DomainHostPtr<T> _domain {};
    CodecHostPtr<T> _codec {};
    SpatialHashingSearcherHostPtr<T> _searcher {};
    ParticleDataHostPtr<T> _particle_data;

    ParticleDeviceProbe<T> _particle_probe {};
    DomainDeviceProbe<T> _domain_probe {};
    SpatialHashingProbe<T> _searcher_probe {};
    CodecDeviceProbe<T> _codec_probe {};

    HostBuffer<SourceHostPtr<T>> _sources;
    HostBuffer<SinkHostPtr<T>> _sinks;
    HostBuffer<ColliderHostPtr<T>> _colliders;
};

template <typename T>
class System<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const DomainHostPtr<T>& domain);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(const CodecHostPtr<T>& codec);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sources(const HostBuffer<SourceHostPtr<T>>& sources);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sinks(const HostBuffer<SinkHostPtr<T>>& sinks);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_colliders(const HostBuffer<Collider<T>>& colliders);

    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _dt { static_cast<T>(0.01) };
    size_t _buffer_size { 0 };

    DomainHostPtr<T> _domain {};
    CodecHostPtr<T> _codec {};

    HostBuffer<SourceHostPtr<T>> _sources;
    HostBuffer<SinkHostPtr<T>> _sinks;
    HostBuffer<ColliderHostPtr<T>> _colliders;
};

}

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>
