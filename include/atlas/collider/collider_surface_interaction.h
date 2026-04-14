#pragma once

/**
 * @file collider_surface_interaction.h
 * @brief Declares particle-surface interaction policies used by collider boundary responses.
 *
 * @details
 * This header defines @ref atlas::system::ColliderSurfaceInteraction, a compact
 * policy object that transforms an incident particle velocity into an outgoing
 * velocity after contact with a collider surface.
 *
 * The interaction model supports a mixture of:
 * - **specular reflection**, where the outgoing direction mirrors the incident
 *   direction about the surface normal,
 * - **diffuse re-emission**, where the outgoing direction is sampled from the
 *   outward hemisphere,
 * - **restitution scaling**, which modifies the final speed magnitude,
 * - **tangential momentum accommodation**, which blends between ideal specular
 *   and diffuse boundary behavior.
 *
 * Such a model is useful in particle-based simulations where wall interactions
 * cannot be treated as a single perfectly elastic rule. Depending on the chosen
 * parameters, the same interaction object can approximate:
 * - nearly elastic mirror-like reflection,
 * - strongly accommodating diffuse wall scattering,
 * - intermediate mixed-response boundary behavior.
 *
 * ## Diffuse scattering law
 * When diffuse re-emission contributes to the outgoing response, the sampled
 * hemisphere direction may follow one of multiple laws:
 * - @ref DiffuseSampling::Uniform, which samples the outward hemisphere uniformly,
 * - @ref DiffuseSampling::CosineWeighted, which biases samples toward the normal
 *   direction and is often more physically relevant for Lambertian-like emission.
 *
 * ## Deterministic pseudo-randomness
 * To avoid reliance on shared random-number-generator state in device code,
 * the implementation exposes a deterministic helper,
 * @ref ColliderSurfaceInteraction::hashed_unit_interval, which derives a
 * repeatable pseudo-random value in \f$[0,1)\f$ from geometric or kinematic seeds.
 *
 * ## Construction
 * The interaction policy may be:
 * - default-constructed with physically reasonable baseline parameters, or
 * - configured through the nested fluent @ref Builder, which stages parameters,
 *   validates them, and builds either a value instance or a `host_shared_ptr`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for velocities, coefficients, and temperature.
 */

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Selects the hemisphere sampling law used for diffuse scattering.
 *
 * @details
 * This enum controls how the outgoing diffuse direction is sampled when the
 * interaction model includes a diffuse re-emission component.
 *
 * Different sampling laws imply different angular distributions:
 * - uniform hemisphere sampling gives equal probability density over solid angle,
 * - cosine-weighted sampling favors directions closer to the surface normal.
 */
enum class DiffuseSampling {
    CosineWeighted, ///< Sample the outward hemisphere with cosine weighting, biasing directions toward the normal.
    Uniform         ///< Sample the outward hemisphere uniformly in direction.
};

/**
 * @brief Encapsulates the boundary response for a particle hitting a surface.
 *
 * @details
 * @ref ColliderSurfaceInteraction models how a particle velocity changes after
 * colliding with a solid boundary.
 *
 * The outgoing response is conceptually built from three ingredients:
 * - a **specular component**, representing ideal mirror reflection,
 * - a **diffuse component**, representing stochastic hemisphere re-emission,
 * - a **restitution factor**, scaling the final outgoing speed.
 *
 * The relative amount of specular versus diffuse behavior is controlled by the
 * tangential momentum accommodation coefficient (TMAC):
 * - `TMAC = 0` typically corresponds to purely specular behavior,
 * - `TMAC = 1` typically corresponds to fully diffuse behavior,
 * - intermediate values produce blended responses.
 *
 * ## Temperature metadata
 * The interaction also stores a wall temperature value. Depending on the current
 * or future implementation, this may be used to:
 * - annotate the physical state of the wall,
 * - influence diffuse re-emission behavior,
 * - provide thermodynamic boundary metadata for more advanced models.
 *
 * ## Host/device usage
 * - Parameter configuration is host-side.
 * - The actual velocity transformation operator is marked `ATLAS_ALL_DEVICE`
 *   so that it can be used in device kernels as well as host code.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class ColliderSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "ColliderSurfaceInteraction requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing
     *        @ref ColliderSurfaceInteraction.
     *
     * @details
     * The builder stages interaction parameters such as restitution, diffuse
     * sampling mode, accommodation, and temperature, then validates them before
     * constructing a final interaction object.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an interaction policy with default-initialized parameters:
     * - restitution = 1
     * - TMAC = 1
     * - temperature = 273.15
     * - diffuse sampling = @ref DiffuseSampling::Uniform
     *
     * These defaults correspond to a fully diffuse interaction model with no
     * speed loss and a nominal wall temperature.
     */
    ColliderSurfaceInteraction() = default;

    /**
     * @brief Default destructor.
     *
     * @details
     * Since this type is a lightweight value object with no custom resource
     * ownership, the destructor is defaulted.
     */
    ~ColliderSurfaceInteraction() = default;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder for fluent configuration.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Set the diffuse hemisphere sampling law.
     *
     * @details
     * Controls how outgoing directions are sampled when the interaction includes
     * a diffuse scattering component.
     *
     * @param mode Diffuse sampling policy to use.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Set the restitution coefficient applied to the outgoing speed.
     *
     * @details
     * The restitution coefficient scales the magnitude of the final outgoing
     * velocity after the reflected/scattered direction has been chosen.
     *
     * Typical interpretations are:
     * - `1` : no speed loss due to the collision,
     * - `0 < e < 1` : partial kinetic energy loss,
     * - `0` : fully damped outgoing speed.
     *
     * @param restitution_coeff Restitution coefficient.
     *
     * @note
     * The physically admissible range is typically implementation-validated and
     * is commonly expected to lie in \f$[0,1]\f$.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_restitution(T restitution_coeff) noexcept;

    /**
     * @brief Set the tangential momentum accommodation coefficient (TMAC).
     *
     * @details
     * TMAC controls the blend between specular reflection and diffuse re-emission.
     *
     * Typical interpretation:
     * - `0`   : purely specular response,
     * - `1`   : purely diffuse response,
     * - between `0` and `1` : mixed interaction.
     *
     * @param tmac Tangential momentum accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tangential_momentum_accommodation(T tmac) noexcept;

    /**
     * @brief Set the wall temperature associated with the interaction.
     *
     * @details
     * Stores the wall/surface temperature metadata used by the interaction model.
     * In more advanced boundary models, this value may influence re-emission
     * statistics or energy accommodation behavior.
     *
     * @param temperature Surface temperature.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    /**
     * @brief Return the active diffuse sampling law.
     *
     * @return Current diffuse sampling mode.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
    diffuse_sampling() const noexcept;

    /**
     * @brief Return the restitution coefficient.
     *
     * @return Current restitution coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    restitution() const noexcept;

    /**
     * @brief Return the tangential momentum accommodation coefficient.
     *
     * @return Current TMAC value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tangential_momentum_accommodation() const noexcept;

    /**
     * @brief Return the wall temperature.
     *
     * @return Current surface temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    /**
     * @brief Compute the outgoing velocity after a particle hits a surface.
     *
     * @details
     * Transforms an incident particle velocity into an outgoing velocity using
     * the interaction policy stored in this object.
     *
     * The response generally involves:
     * - determining the specular reflection direction from the incident velocity
     *   and outward normal,
     * - optionally sampling a diffuse outgoing hemisphere direction,
     * - blending those responses using TMAC,
     * - scaling the final magnitude using the restitution coefficient.
     *
     * @param incident Incoming particle velocity before collision.
     * @param normal Outward-facing surface normal at the collision point.
     * @return Outgoing particle velocity after reflection/scattering.
     *
     * @note
     * For physically meaningful results, @p normal is typically expected to be
     * normalized. The precise assumptions are implementation-defined in
     * `collider_surface_interaction.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

private:
    /**
     * @brief Speed scaling applied after the outgoing direction is chosen.
     *
     * @details
     * This coefficient acts as the restitution factor of the boundary response.
     */
    T _restitution_coeff { T(1) };

    /**
     * @brief Blend factor between specular and diffuse response.
     *
     * @details
     * Interpreted as the tangential momentum accommodation coefficient (TMAC).
     * Lower values favor specular reflection; higher values favor diffuse
     * re-emission.
     */
    T _tmac { T(1) };

    /**
     * @brief Surface temperature metadata.
     *
     * @details
     * Stores the wall temperature associated with the interaction model.
     */
    T _temperature { T(273.15) };

    /**
     * @brief Diffuse hemisphere sampling mode.
     *
     * @details
     * Controls the angular distribution used when generating diffuse outgoing
     * directions.
     */
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
};

/**
 * @brief Fluent builder for @ref ColliderSurfaceInteraction.
 *
 * @details
 * The builder provides a controlled way to construct a
 * @ref ColliderSurfaceInteraction while staging and validating the full set of
 * interaction parameters.
 *
 * ## Typical usage
 * @code
 * auto interaction = atlas::ColliderSurfaceInteraction<float>::builder()
 *     .with_diffuse_sampling(atlas::system::DiffuseSampling::CosineWeighted)
 *     .with_restitution(0.9f)
 *     .with_tangential_momentum_accommodation(0.7f)
 *     .with_temperature(300.0f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. A typical
 * validation policy may check:
 * - restitution is within an admissible range,
 * - TMAC is within an admissible range,
 * - temperature is finite and physically acceptable.
 *
 * The exact validation rules are implementation-defined in
 * `collider_surface_interaction.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class ColliderSurfaceInteraction<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged interaction parameters to the same defaults as the
     * final interaction object.
     */
    Builder() = default;

    /**
     * @brief Set the diffuse hemisphere sampling law.
     *
     * @param mode Diffuse sampling policy to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Set the restitution coefficient.
     *
     * @details
     * Stages the outgoing speed scaling factor to be stored in the final
     * interaction object.
     *
     * @param restitution Restitution coefficient.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_restitution(T restitution) noexcept;

    /**
     * @brief Set the tangential momentum accommodation coefficient.
     *
     * @details
     * Stages the blend parameter that controls the balance between specular and
     * diffuse boundary response.
     *
     * @param tmac Tangential momentum accommodation coefficient.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tangential_momentum_accommodation(T tmac) noexcept;

    /**
     * @brief Set the surface temperature.
     *
     * @details
     * Stages the wall temperature metadata for the final interaction object.
     *
     * @param temperature Surface temperature.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    /**
     * @brief Build a configured @ref ColliderSurfaceInteraction by value.
     *
     * @details
     * Validates the currently staged parameters and constructs the final
     * interaction object by value.
     *
     * @return Constructed interaction object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ColliderSurfaceInteraction<T>
    build() const;

    /**
     * @brief Build a configured @ref ColliderSurfaceInteraction in a host_shared_ptr.
     *
     * @details
     * Validates the currently staged parameters, constructs the interaction
     * object, and returns it in a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>` owning the constructed object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate staged interaction parameters.
     *
     * @details
     * Performs pre-construction checks on the builder state.
     *
     * Typical checks may include:
     * - restitution lies in a valid range,
     * - TMAC lies in a valid range,
     * - temperature is finite and acceptable.
     *
     * @note
     * The exact validation policy is implementation-defined in
     * `collider_surface_interaction.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending diffuse hemisphere sampling mode.
     */
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };

    /**
     * @brief Pending restitution coefficient.
     */
    T _restitution { T(1) };

    /**
     * @brief Pending tangential momentum accommodation coefficient.
     */
    T _tmac { T(1) };

    /**
     * @brief Pending wall temperature.
     */
    T _temperature { T(273.15) };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::ColliderSurfaceInteraction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderSurfaceInteraction = atlas::system::ColliderSurfaceInteraction<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::ColliderSurfaceInteraction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderSurfaceInteractionHostPtr = atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::ColliderSurfaceInteraction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderSurfaceInteractionDevicePtr = atlas::device_shared_ptr<ColliderSurfaceInteraction<T>>;

} // namespace atlas

#include <atlas/collider/collider_surface_interaction.hpp>