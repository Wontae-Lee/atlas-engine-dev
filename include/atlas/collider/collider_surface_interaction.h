#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

enum class DiffuseSampling {
    CosineWeighted,
    Uniform
};

template <typename T>
class ColliderSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "ColliderSurfaceInteraction requires a floating-point T");

public:
    class Builder;

public:
    ColliderSurfaceInteraction()  = default;
    ~ColliderSurfaceInteraction() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_restitution(T restitution_coeff) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tangential_momentum_accommodation(T tmac) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
    diffuse_sampling() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    restitution() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tangential_momentum_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    hashed_unit_interval(const Vector3<T>& seed, T salt) noexcept;

private:
    T _restitution_coeff { T(1) };
    T _tmac { T(1) };
    T _temperature { T(273.15) };
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
};

template <typename T>
class ColliderSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_restitution(T restitution) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tangential_momentum_accommodation(T tmac) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ColliderSurfaceInteraction<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
    T _restitution { T(1) };
    T _tmac { T(1) };
    T _temperature { T(273.15) };
};

}

namespace atlas {

template <typename T>
using ColliderSurfaceInteraction = atlas::system::ColliderSurfaceInteraction<T>;

template <typename T>
using ColliderSurfaceInteractionHostPtr = atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>;

template <typename T>
using ColliderSurfaceInteractionDevicePtr = atlas::device_shared_ptr<ColliderSurfaceInteraction<T>>;

}

#include <atlas/collider/collider_surface_interaction.hpp>
