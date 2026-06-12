#pragma once

/**
 * @file maxwellian_surface_interaction.h
 * @brief Declares a SPARTA-compatible Maxwellian diffuse surface interaction model.
 */

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas {

/**
 * @brief SPARTA-style internal-energy sampling mode for wall collisions.
 */
enum struct MaxwellianInternalEnergyStyle : int {
    none,
    smooth,
    discrete
};

/**
 * @brief Maxwellian diffuse/specular wall interaction matching SPARTA diffuse surfaces.
 *
 * The diffuse branch follows SPARTA's `surf_collide diffuse` velocity sampling:
 * - normal velocity is sampled from @f$vrm \sqrt{-\log(U)}@f$
 * - tangential speed is sampled from @f$vrm \sqrt{-\log(U)}@f$
 * - tangential angle is sampled uniformly in @f$[0, 2\pi)@f$
 *
 * The momentum accommodation coefficient is interpreted like SPARTA diffuse
 * `acc`: values in [0, 1] choose diffuse scattering with probability
 * `momentum_acc`, and specular reflection otherwise.
 *
 * The translational, rotational, and vibrational accommodation coefficients are
 * stored with the same naming used by SPARTA's extended surface models so the
 * same payload can carry all wall accommodation parameters.
 *
 * @tparam T Floating-point scalar type used for all computations.
 */
template <typename T>
class MaxwellianSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "MaxwellianSurfaceInteraction requires a floating-point T");

public:
    /**
     * @brief Builder for configuring MaxwellianSurfaceInteraction objects.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Initializes the interaction with:
     * - temperature = 273.15 K
     * - molecular mass = 1 kg
     * - momentum accommodation = 1
     * - translational accommodation = 1
     * - rotational accommodation = 1
     * - vibrational accommodation = 1
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MaxwellianSurfaceInteraction() noexcept = default;

    /**
     * @brief Destructor.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    ~MaxwellianSurfaceInteraction() noexcept = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent interaction configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Sets the wall temperature in Kelvin.
     *
     * @param temperature Surface temperature.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    /**
     * @brief Sets the particle molecular mass in kilograms.
     *
     * @param molecular_mass Particle molecular mass.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_molecular_mass(T molecular_mass) noexcept;

    /**
     * @brief Sets the momentum accommodation coefficient.
     *
     * @param momentum_acc Diffuse scattering probability in [0, 1].
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_momentum_acc(T momentum_acc) noexcept;

    /**
     * @brief Sets the translational accommodation coefficient.
     *
     * @param trans_acc Translational accommodation coefficient in [0, 1].
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_trans_acc(T trans_acc) noexcept;

    /**
     * @brief Sets the rotational accommodation coefficient.
     *
     * @param rot_acc Rotational accommodation coefficient in [0, 1].
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_rot_acc(T rot_acc) noexcept;

    /**
     * @brief Sets the vibrational accommodation coefficient.
     *
     * @param vib_acc Vibrational accommodation coefficient in [0, 1].
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vib_acc(T vib_acc) noexcept;

    /**
     * @brief Sets the rotational internal-energy sampling style.
     *
     * @param style SPARTA-style rotational sampling mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    /**
     * @brief Sets the vibrational internal-energy sampling style.
     *
     * @param style SPARTA-style vibrational sampling mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    /**
     * @brief Sets all SPARTA-style accommodation coefficients.
     *
     * @param momentum_acc Momentum accommodation coefficient in [0, 1].
     * @param trans_acc Translational accommodation coefficient in [0, 1].
     * @param rot_acc Rotational accommodation coefficient in [0, 1].
     * @param vib_acc Vibrational accommodation coefficient in [0, 1].
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_accommodation(T momentum_acc, T trans_acc, T rot_acc, T vib_acc) noexcept;

    /**
     * @brief Returns the wall temperature.
     *
     * @return Surface temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    /**
     * @brief Returns the configured molecular mass.
     *
     * @return Particle molecular mass.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    molecular_mass() const noexcept;

    /**
     * @brief Returns the momentum accommodation coefficient.
     *
     * @return Diffuse scattering probability.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    momentum_acc() const noexcept;

    /**
     * @brief Returns the translational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    trans_acc() const noexcept;

    /**
     * @brief Returns the rotational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    rot_acc() const noexcept;

    /**
     * @brief Returns the vibrational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    vib_acc() const noexcept;

    /**
     * @brief Returns the rotational internal-energy sampling style.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE MaxwellianInternalEnergyStyle
    rot_style() const noexcept;

    /**
     * @brief Returns the vibrational internal-energy sampling style.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE MaxwellianInternalEnergyStyle
    vib_style() const noexcept;

    /**
     * @brief Returns SPARTA's most probable molecular speed scale.
     *
     * @return @f$sqrt(2 k_B T / m)@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    most_probable_speed() const noexcept;

    /**
     * @brief Computes a deterministic post-collision velocity.
     *
     * Uniform samples are derived from the incident velocity and normal through
     * Atlas hash sampling, then passed to @ref sample.
     *
     * @param incident Incident velocity.
     * @param normal Outward unit surface normal.
     * @return Post-collision velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    /**
     * @brief Computes post-collision velocity from explicit uniform samples.
     *
     * This method exists so benchmarks and tests can inject the exact same RNG
     * stream as a reference SPARTA implementation and compare algorithmic output
     * without coupling the model to SPARTA's mutable RNG state.
     *
     * @param incident Incident velocity.
     * @param normal Outward unit surface normal.
     * @param branch_sample Uniform sample for diffuse/specular branch selection.
     * @param perpendicular_sample Uniform sample for normal speed.
     * @param theta_sample Uniform sample for tangential angle.
     * @param tangent_sample Uniform sample for tangential speed.
     * @param tangent_seed Fallback vector used for normal-incidence tangent construction.
     * @return Post-collision velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sample(const Vector3<T>& incident,
           const Vector3<T>& normal,
           T branch_sample,
           T perpendicular_sample,
           T theta_sample,
           T tangent_sample,
           const Vector3<T>& tangent_seed) const noexcept;

    /**
     * @brief Computes deterministic post-collision internal energy.
     *
     * Uniform samples are derived from the incident internal-energy state and
     * then passed to @ref sample_internal_energy.
     *
     * @param incident Incident per-particle internal energy.
     * @return Post-collision internal energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident) const noexcept;

    /**
     * @brief Computes deterministic post-collision internal energy for a surface hit.
     *
     * The diffuse/specular branch uses the same deterministic branch sample as
     * the velocity collision so specular collisions preserve internal energy.
     *
     * @param incident_energy Incident per-particle internal energy.
     * @param incident_velocity Incident velocity relative to the surface.
     * @param normal Outward unit surface normal.
     * @return Post-collision internal energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal) const noexcept;

    /**
     * @brief Computes SPARTA diffuse post-collision internal energy for a surface hit.
     *
     * The diffuse/specular branch uses the same deterministic branch sample as
     * velocity scattering. Specular collisions preserve internal energy, while
     * diffuse collisions resample rotational and vibrational energy from the
     * wall temperature using species material data.
     *
     * @param incident_energy Incident per-particle internal energy.
     * @param incident_velocity Incident velocity relative to the surface.
     * @param normal Outward unit surface normal.
     * @param material Species material properties for the colliding particle.
     * @return Post-collision internal energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal,
                    const MaterialProperties<T>& material) const noexcept;

    /**
     * @brief Computes post-collision internal energy from explicit uniform samples.
     *
     * The rotational and vibrational branches follow SPARTA CLL's continuous
     * two-degree-of-freedom accommodation form. The translational scalar is
     * treated with the same energy accommodation law so the stored state can
     * carry a wall-accommodated translational energy when a caller keeps one.
     *
     * @param incident Incident per-particle internal energy.
     * @param trans_sample Uniform sample for translational energy magnitude.
     * @param trans_theta_sample Uniform sample for translational energy phase.
     * @param rot_sample Uniform sample for rotational energy magnitude.
     * @param rot_theta_sample Uniform sample for rotational energy phase.
     * @param vib_sample Uniform sample for vibrational energy magnitude.
     * @param vib_theta_sample Uniform sample for vibrational energy phase.
     * @return Post-collision internal energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    sample_internal_energy(const FluidInternalEnergy<T>& incident,
                           T trans_sample,
                           T trans_theta_sample,
                           T rot_sample,
                           T rot_theta_sample,
                           T vib_sample,
                           T vib_theta_sample) const noexcept;

    /**
     * @brief Applies wall accommodation to one scalar internal-energy mode.
     *
     * @param incident Incident mode energy.
     * @param acc Accommodation coefficient in [0, 1].
     * @param sample Uniform sample for wall energy magnitude.
     * @param theta_sample Uniform sample for phase coupling.
     * @return Post-collision mode energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_internal_energy_mode(T incident, T acc, T sample, T theta_sample) const noexcept;

    /**
     * @brief Samples SPARTA diffuse rotational wall energy.
     *
     * @param material Species material properties.
     * @param seed Deterministic random seed.
     * @return Sampled rotational energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_rotational_energy(const MaterialProperties<T>& material,
                                     const Vector3<T>& seed) const noexcept;

    /**
     * @brief Samples SPARTA diffuse vibrational wall energy.
     *
     * @param material Species material properties.
     * @param seed Deterministic random seed.
     * @return Sampled vibrational energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_vibrational_energy(const MaterialProperties<T>& material,
                                      const Vector3<T>& seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_smooth_energy(int dof, const Vector3<T>& seed, T salt) const noexcept;

private:
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    T _momentum_acc { T(1) };
    T _trans_acc { T(1) };
    T _rot_acc { T(1) };
    T _vib_acc { T(1) };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

template <typename T>
class MaxwellianSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    /**
     * @brief Sets the wall temperature in Kelvin.
     *
     * @param temperature Surface temperature.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    /**
     * @brief Sets the particle molecular mass in kilograms.
     *
     * @param molecular_mass Particle molecular mass.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    /**
     * @brief Sets the momentum accommodation coefficient.
     *
     * @param momentum_acc Diffuse scattering probability in [0, 1].
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_momentum_acc(T momentum_acc) noexcept;

    /**
     * @brief Sets the translational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_trans_acc(T trans_acc) noexcept;

    /**
     * @brief Sets the rotational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rot_acc(T rot_acc) noexcept;

    /**
     * @brief Sets the vibrational accommodation coefficient.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vib_acc(T vib_acc) noexcept;

    /**
     * @brief Sets the rotational internal-energy sampling style.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    /**
     * @brief Sets the vibrational internal-energy sampling style.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    /**
     * @brief Sets all SPARTA-style accommodation coefficients.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_accommodation(T momentum_acc, T trans_acc, T rot_acc, T vib_acc) noexcept;

    /**
     * @brief Builds a validated MaxwellianSurfaceInteraction object.
     *
     * @return Constructed interaction object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellianSurfaceInteraction<T>
    build() const;

    /**
     * @brief Builds a host-side shared MaxwellianSurfaceInteraction instance.
     *
     * @return Host shared pointer to the constructed interaction object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellianSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    T _momentum_acc { T(1) };
    T _trans_acc { T(1) };
    T _rot_acc { T(1) };
    T _vib_acc { T(1) };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

} // namespace atlas

namespace atlas {
template <typename T>
using MaxwellianSurfaceInteractionHostPtr = atlas::host_shared_ptr<MaxwellianSurfaceInteraction<T>>;

template <typename T>
using MaxwellianSurfaceInteractionDevicePtr = atlas::device_shared_ptr<MaxwellianSurfaceInteraction<T>>;

} // namespace atlas

#include <atlas/collider/interaction/maxwellian_surface_interaction.hpp>
