#pragma once

/**
 * @file collider_surface_interaction.h
 * @brief Declares particle-surface interaction policies for colliders.
 *
 * ColliderSurfaceInteraction controls how a particle velocity is transformed
 * after striking a collider surface. It supports specular reflection, diffuse
 * hemisphere sampling, restitution scaling, and tangential momentum
 * accommodation blending.
 */

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Selects the sampling law used for diffuse scattering.
 */
enum class DiffuseSampling {
    CosineWeighted, ///< Bias diffuse samples toward the surface normal.
    Uniform         ///< Sample the outgoing hemisphere uniformly.
};

/**
 * @brief Encapsulates the boundary response for a particle hitting a surface.
 *
 * The interaction can interpolate between specular reflection and diffuse
 * re-emission using a tangential momentum accommodation coefficient. The final
 * outgoing speed is scaled by a restitution coefficient.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class ColliderSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "ColliderSurfaceInteraction requires a floating-point T");

public:
    /**
     * @brief Fluent builder for ColliderSurfaceInteraction.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    ColliderSurfaceInteraction()  = default;
    /**
     * @brief Destructor.
     */
    ~ColliderSurfaceInteraction() = default;

    /**
     * @brief Returns a builder initialized with default values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Sets the diffuse sampling law.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Sets the restitution coefficient applied after reflection/scattering.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_restitution(T restitution_coeff) noexcept;

    /**
     * @brief Sets the tangential momentum accommodation coefficient.
     *
     * A value near 0 favors specular response, while a value near 1 favors
     * diffuse re-emission.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tangential_momentum_accommodation(T tmac) noexcept;

    /**
     * @brief Sets the wall temperature associated with the interaction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    /**
     * @brief Returns the diffuse sampling law.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
    diffuse_sampling() const noexcept;

    /**
     * @brief Returns the restitution coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    restitution() const noexcept;

    /**
     * @brief Returns the tangential momentum accommodation coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tangential_momentum_accommodation() const noexcept;

    /**
     * @brief Returns the wall temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    /**
     * @brief Computes the outgoing velocity after a surface hit.
     *
     * @param incident Incoming particle velocity.
     * @param normal Outward-facing surface normal at the hit point.
     * @return Vector3<T> Reflected or diffusely scattered outgoing velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    /**
     * @brief Generates a deterministic pseudo-random value in [0, 1).
     *
     * This helper avoids shared RNG state and derives repeatable scattering
     * randomness directly from geometric/velocity seeds.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    hashed_unit_interval(const Vector3<T>& seed, T salt) noexcept;

private:
    T _restitution_coeff { T(1) }; ///< Speed scaling applied to the outgoing direction.
    T _tmac { T(1) }; ///< Blend between specular and diffuse response.
    T _temperature { T(273.15) }; ///< Surface temperature metadata.
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform }; ///< Diffuse hemisphere sampling mode.
};

template <typename T>
class ColliderSurfaceInteraction<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the diffuse sampling law.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Sets the restitution coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_restitution(T restitution) noexcept;

    /**
     * @brief Sets the tangential momentum accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tangential_momentum_accommodation(T tmac) noexcept;

    /**
     * @brief Sets the surface temperature.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    /**
     * @brief Builds a value instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ColliderSurfaceInteraction<T>
    build() const;

    /**
     * @brief Builds a host-shared interaction instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates builder state before construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform }; ///< Pending diffuse sampling mode.
    T _restitution { T(1) }; ///< Pending restitution coefficient.
    T _tmac { T(1) }; ///< Pending tangential momentum accommodation.
    T _temperature { T(273.15) }; ///< Pending wall temperature.
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::ColliderSurfaceInteraction.
 */
template <typename T>
using ColliderSurfaceInteraction = atlas::system::ColliderSurfaceInteraction<T>;

/**
 * @brief Host shared pointer alias for ColliderSurfaceInteraction.
 */
template <typename T>
using ColliderSurfaceInteractionHostPtr = atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>;

/**
 * @brief Device shared pointer alias for ColliderSurfaceInteraction.
 */
template <typename T>
using ColliderSurfaceInteractionDevicePtr = atlas::device_shared_ptr<ColliderSurfaceInteraction<T>>;

}

#include <atlas/collider/collider_surface_interaction.hpp>
