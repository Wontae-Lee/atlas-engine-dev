#pragma once

/**
 * @file isothermal_surface_kernel.h
 * @brief Declares a configurable isothermal interaction for surface reflections.
 */

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas {

/**
 * @brief Diffuse hemisphere sampling mode used for surface scattering.
 *
 * This enumeration selects how a diffuse outgoing direction is sampled when
 * the interaction model chooses a diffuse reflection event.
 */
enum class DiffuseSampling {
    /**
     * @brief Cosine-weighted hemisphere sampling.
     *
     * Directions closer to the surface normal are sampled more frequently.
     */
    CosineWeighted,

    /**
     * @brief Uniform hemisphere sampling.
     *
     * All directions over the hemisphere are sampled with equal probability.
     */
    Uniform
};

/**
 * @brief Surface interaction model for isothermal surface reflections.
 *
 * This class describes how an incident velocity interacts with a surface
 * normal to produce an outgoing velocity. The model supports:
 * - purely specular reflection
 * - purely diffuse reflection
 * - stochastic mixing between specular and diffuse reflection
 *
 * The mixing behavior is controlled by the momentum accommodation
 * coefficient. Additional parameters such as restitution and temperature
 * are stored as part of the interaction configuration.
 *
 * @tparam T Floating-point scalar type used for all computations.
 */
template <typename T>
class IsothermalSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "IsothermalSurfaceInteraction requires a floating-point T");

public:
    /**
     * @brief Builder for configuring and constructing IsothermalSurfaceInteraction objects.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Initializes the interaction with default parameters:
     * - restitution = 1
     * - momentum accommodation = 1
     * - temperature = 273.15
     * - diffuse sampling = Uniform
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalSurfaceInteraction() noexcept = default;

    /**
     * @brief Destructor.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    ~IsothermalSurfaceInteraction() noexcept = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent interaction configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Sets the diffuse sampling mode.
     *
     * This setting affects how diffuse outgoing directions are sampled when the
     * interaction selects the diffuse branch.
     *
     * @param mode Diffuse hemisphere sampling mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Sets the restitution coefficient.
     *
     * The restitution coefficient scales the outgoing speed relative to the
     * incident speed:
     * - `0` removes all reflected speed,
     * - `1` preserves the reflected speed magnitude.
     *
     * @param restitution_coeff Restitution coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_restitution(T restitution_coeff) noexcept;

    /**
     * @brief Sets the momentum accommodation coefficient.
     *
     * In the current model:
     * - 0 corresponds to purely specular reflection
     * - 1 corresponds to purely diffuse reflection
     * - intermediate values produce stochastic mixing
     *
     * @param momentum_acc Momentum accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_momentum_acc(T momentum_acc) noexcept;

    /**
     * @brief Sets the surface temperature parameter.
     *
     * This parameter is stored as part of the interaction model and may be used
     * by thermal or extended interaction formulations.
     *
     * @param temperature Surface temperature.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    /**
     * @brief Returns the configured diffuse sampling mode.
     *
     * @return Diffuse hemisphere sampling mode.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
    diffuse_sampling() const noexcept;

    /**
     * @brief Returns the restitution coefficient.
     *
     * @return Restitution coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    restitution() const noexcept;

    /**
     * @brief Returns the momentum accommodation coefficient.
     *
     * @return Momentum accommodation value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    momentum_acc() const noexcept;

    /**
     * @brief Returns the stored surface temperature.
     *
     * @return Surface temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    /**
     * @brief Computes the outgoing velocity resulting from surface interaction.
     *
     * Given an incident velocity and a surface normal, this operator produces
     * an outgoing velocity according to the configured reflection model.
     *
     * The exact behavior depends on the current interaction parameters such as
     * momentum accommodation, restitution, and diffuse sampling mode.
     *
     * The outgoing speed is `incident.length() * restitution()`, while the
     * outgoing direction is selected by the configured specular/diffuse model.
     *
     * @param incident Incident velocity-like vector.
     * @param normal Surface normal defining the reflection hemisphere.
     * @return Outgoing velocity-like vector after interaction.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

private:
    /**
     * @brief Coefficient applied to scale outgoing speed after reflection.
     */
    T _restitution_coeff { T(1) };

    /**
     * @brief Momentum accommodation coefficient controlling diffuse/specular mixing.
     */
    T _momentum_acc { T(1) };

    /**
     * @brief Surface temperature associated with this interaction model.
     */
    T _temperature { T(273.15) };

    /**
     * @brief Sampling mode used when generating diffuse outgoing directions.
     */
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
};

/**
 * @brief Builder for IsothermalSurfaceInteraction.
 *
 * Provides a fluent interface for configuring interaction parameters before
 * constructing a validated IsothermalSurfaceInteraction instance.
 *
 * @tparam T Floating-point scalar type used for all computations.
 */
template <typename T>
class IsothermalSurfaceInteraction<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the diffuse sampling mode for the interaction being built.
     *
     * @param mode Diffuse hemisphere sampling mode.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Sets the restitution coefficient for the interaction being built.
     *
     * @param restitution Restitution coefficient.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_restitution(T restitution) noexcept;

    /**
     * @brief Sets the momentum accommodation coefficient.
     *
     * @param momentum_acc Momentum accommodation coefficient.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_momentum_acc(T momentum_acc) noexcept;

    /**
     * @brief Sets the surface temperature parameter.
     *
     * @param temperature Surface temperature.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    /**
     * @brief Builds a validated IsothermalSurfaceInteraction object.
     *
     * @return Constructed interaction object.
     *
     * @throw std::runtime_error Thrown if the configured parameters are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE IsothermalSurfaceInteraction<T>
    build() const;

    /**
     * @brief Builds a host-side shared instance of IsothermalSurfaceInteraction.
     *
     * @return Host shared pointer to the constructed interaction object.
     *
     * @throw std::runtime_error Thrown if the configured parameters are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<IsothermalSurfaceInteraction<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the current builder configuration.
     *
     * Ensures that all physical parameters are finite and within the supported
     * ranges required by the interaction model.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Diffuse hemisphere sampling mode to apply in the built interaction.
     */
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };

    /**
     * @brief Restitution coefficient for the built interaction.
     */
    T _restitution { T(1) };

    /**
     * @brief Momentum accommodation value controlling diffuse/specular mixing.
     */
    T _momentum_acc { T(1) };

    /**
     * @brief Surface temperature for the built interaction.
     */
    T _temperature { T(273.15) };
};

} // namespace atlas

namespace atlas {

/**
 * @brief Host-side shared pointer alias for IsothermalSurfaceInteraction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using IsothermalSurfaceInteractionHostPtr = atlas::host_shared_ptr<IsothermalSurfaceInteraction<T>>;

/**
 * @brief Device-side shared pointer alias for IsothermalSurfaceInteraction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using IsothermalSurfaceInteractionDevicePtr = atlas::device_shared_ptr<IsothermalSurfaceInteraction<T>>;

} // namespace atlas

#include <atlas/collider/interaction/isothermal_surface_kernel.hpp>
