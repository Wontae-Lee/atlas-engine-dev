#pragma once

#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    class Builder;

public:
    Collider()  = default;
    ~Collider() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(UnitHostPtr<T> unit,
             atlas::host_shared_ptr<ColliderSurfaceInteraction<T>> surface_interaction) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(const UnitHostPtr<T>& unit);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_surface_interaction(const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>&
                                surface_interaction);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const UnitHostPtr<T>&
    unit() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>&
    surface_interaction() const noexcept;

private:
    UnitHostPtr<T> _unit                                                       = nullptr;
    atlas::host_shared_ptr<ColliderSurfaceInteraction<T>> _surface_interaction = nullptr;
};

template <typename T>
class Collider<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(const UnitHostPtr<T>& unit);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interaction(const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>&
                                 surface_interaction);

    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UnitHostPtr<T> _unit                                                       = nullptr;
    atlas::host_shared_ptr<ColliderSurfaceInteraction<T>> _surface_interaction = nullptr;
};

}

namespace atlas {

template <typename T>
using Collider = atlas::system::Collider<T>;

template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

}

#include <atlas/collider/collider.hpp>
