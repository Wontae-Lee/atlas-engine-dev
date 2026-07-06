#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>

/**
 * @file isothermal_surface_kernel.h
 * @brief Restitution-based gas/granular-surface interaction: preserves
 *        incident speed (scaled by a coefficient of restitution) and
 *        blends specular and diffuse *direction* by a momentum
 *        accommodation coefficient, without full Maxwellian thermal
 *        speed resampling.
 *
 * @details
 * ### Background
 * `MaxwellianSurfaceInteraction` (see that file) models a molecule fully
 * re-thermalizing to the wall temperature on a diffuse hit — the outgoing
 * *speed* itself is redrawn from the wall's equilibrium distribution.
 * `IsothermalSurfaceInteraction` instead follows the coefficient-of-
 * restitution convention common to granular/discrete-element and
 * macroscopic-particle bounce models: the outgoing speed is always the
 * incident speed scaled by a single `_restitution_coeff` (`1` =
 * perfectly elastic, `<1` = lossy), and only the outgoing *direction* is
 * stochastic — blended between the specular reflection direction and a
 * randomly sampled hemisphere direction by `_momentum_acc`. This is a
 * cheaper, less physically-detailed wall model than the full Maxwellian
 * kernel; it does not exchange translational or internal energy with a
 * wall temperature at all (`internal_energy()` below is a no-op passthrough,
 * and `_temperature` is stored for API symmetry/future use but does not
 * currently affect `operator()`).
 *
 * ### Operating principle
 * `operator()`:
 * 1. Degenerate case: an incident velocity at/below `atlas::tol` returns
 *    the zero vector (nothing to reflect).
 * 2. Computes the specular direction (`atlas::reflected`).
 * 3. If `_momentum_acc <= 0`, always takes the specular direction (purely
 *    elastic-mirror wall) — skipping the random sampling entirely.
 * 4. Otherwise draws a diffuse direction over the hemisphere above
 *    `normal`, either `atlas::sample_cosine_hemisphere` (Lambertian:
 *    outgoing direction density proportional to `cos(theta)` from the
 *    normal, the physically standard diffuse-emission law, same
 *    Lambert's-cosine-law origin cited in
 *    `maxwellian_surface_interaction.h`) or
 *    `atlas::sample_uniform_hemisphere` (density uniform over solid
 *    angle), per `_diffuse_sampling`.
 * 5. A third hashed sample picks between the specular and diffuse
 *    directions with probability `_momentum_acc` of diffuse — the same
 *    Maxwell accommodation-coefficient blend as
 *    `MaxwellianSurfaceInteraction`, applied to direction only.
 * 6. The chosen unit direction is scaled by `incident_speed *
 *    _restitution_coeff` to produce the outgoing velocity.
 */

namespace atlas {

/**
 * @brief Selects the angular distribution of the diffusely-reflected
 *        direction in `IsothermalSurfaceInteraction::operator()`.
 */
enum class DiffuseSampling {

    /** Lambertian: density proportional to `cos(theta)` from the
     *  surface normal (physically standard diffuse emission). */
    cosine_weighted,

    /** Density uniform over the hemisphere's solid angle. */
    uniform
};

/**
 * @brief Restitution + specular/diffuse-direction-blend wall model; see
 *        this file's top-of-file documentation for how it differs from
 *        `MaxwellianSurfaceInteraction` and its sampling procedure.
 */
class IsothermalSurfaceInteraction final {
public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST void
    set_restitution(float restitution_coeff) noexcept;

    ATLAS_HOST void
    set_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST void
    set_temperature(float temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST DiffuseSampling
    diffuse_sampling() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    restitution() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    momentum_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept;

    /**
     * @brief Reflects `incident` off a wall with normal `normal`: fixed
     *        speed scaling by `_restitution_coeff`, direction
     *        stochastically blended between specular and a sampled
     *        diffuse hemisphere direction by `_momentum_acc`. See this
     *        file's top-of-file documentation for the full derivation.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const Float3& incident, const Float3& normal) const noexcept {
        const float incident_speed = incident.length();

        if (incident_speed <= atlas::tol) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const Float3 specular_dir  = atlas::reflected(incident, normal);
        const Float3 specular_unit = specular_dir.normalized();

        if (_momentum_acc <= 0.0f) {
            return specular_unit * (incident_speed * _restitution_coeff);
        }

        const float u1 = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float u2 = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        Float3 diffuse_dir {};

        if (_diffuse_sampling == DiffuseSampling::cosine_weighted) {
            diffuse_dir = atlas::sample_cosine_hemisphere(normal, u1, u2);
        } else {
            diffuse_dir = atlas::sample_uniform_hemisphere(normal, u1, u2);
        }

        const float mix = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const Float3 out_unit = (mix < _momentum_acc)
            ? diffuse_dir.normalized()
            : specular_unit;

        return out_unit * (incident_speed * _restitution_coeff);
    }

    /**
     * @brief No-op passthrough: this wall model does not exchange
     *        internal energy with the wall (unlike
     *        `MaxwellianSurfaceInteraction::internal_energy`), so the
     *        incident value is returned unchanged. Present only to
     *        satisfy the common interface `SurfaceInteractionKernel`
     *        dispatches through.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal,
                    const MaterialProperties& material) const noexcept {
        static_cast<void>(incident_velocity);
        static_cast<void>(normal);
        static_cast<void>(material);
        return incident_energy;
    }

private:
    float _restitution_coeff { 1.0f };

    float _momentum_acc { 1.0f };

    float _temperature { 273.15f };

    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };
};

/**
 * @brief Fluent builder for `IsothermalSurfaceInteraction`; defaults to
 *        a perfectly elastic (`restitution = 1`), fully diffuse
 *        (`momentum_acc = 1`), uniformly-sampled wall.
 */
class IsothermalSurfaceInteraction::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST Builder&
    with_restitution(float restitution) noexcept;

    ATLAS_HOST Builder&
    with_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST IsothermalSurfaceInteraction
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<IsothermalSurfaceInteraction>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };

    float _restitution { 1.0f };

    float _momentum_acc { 1.0f };

    float _temperature { 273.15f };
};

using IsothermalSurfaceInteractionHostPtr = atlas::host_shared_ptr<IsothermalSurfaceInteraction>;

using IsothermalSurfaceInteractionDevicePtr = atlas::device_shared_ptr<IsothermalSurfaceInteraction>;

}
