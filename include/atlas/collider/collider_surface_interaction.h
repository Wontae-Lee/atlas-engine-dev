#ifndef ATLAS_ENGINE_DEV_COLLIDER_SURFACE_INTERACTION_H
#define ATLAS_ENGINE_DEV_COLLIDER_SURFACE_INTERACTION_H

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas {
namespace system {

    enum class DiffuseSampling {
        CosineWeighted,
        Uniform
    };

    template <typename T>
    class ColliderSurfaceInteraction final {
    public:
        ColliderSurfaceInteraction()  = default;
        ~ColliderSurfaceInteraction() = default;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_diffuse_sampling(DiffuseSampling mode);

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_restitution(T restitution_coeff_);

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_tangential_momentum_accommodation(T tmac_);

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
        diffuse_sampling() const;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        restitution() const;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        tangential_momentum_accommodation() const;

        ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        operator()(const Vector3<T>& incident, const Vector3<T>& normal) const;

    private:
        T _restituion_coeff { T(1) };
        T _tmac { T(1) };
        DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
    };

}

template <typename T>
using ColliderSurfaceInteraction = system::ColliderSurfaceInteraction<T>;

template <typename T>
using ColliderSurfaceInteractionHostPtr = host_shared_ptr<ColliderSurfaceInteraction<T>>;

template <typename T>
using ColliderSurfaceInteractionDevicePtr = device_shared_ptr<ColliderSurfaceInteraction<T>>;

}

#include <atlas/collider/collider_surface_interaction.hpp>

#endif