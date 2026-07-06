#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <limits>

/**
 * @file maxwellian_surface_interaction.h
 * @brief Gas-surface interaction model for DSMC wall boundaries, combining
 *        specular and fully-diffuse (Maxwellian) molecular reflection.
 *
 * @details
 * ### Background
 * When a rarefied-gas particle strikes a solid wall, what happens next
 * depends on how the incident molecule exchanges momentum and energy with
 * the surface lattice. Two idealized limits bound the real behaviour:
 *
 * - **Specular reflection**: the wall behaves like a perfect mirror. The
 *   tangential velocity component is preserved and only the normal
 *   component is reversed, so the molecule leaves with the same speed and
 *   an equal angle of reflection. No momentum or energy is exchanged with
 *   the wall.
 * - **Diffuse (Maxwellian) reflection**: the molecule is assumed to
 *   equilibrate with the wall (e.g. via adsorption/re-emission or many
 *   sub-lattice collisions) and leaves with a velocity distribution
 *   identical to that of a gas in equilibrium at the wall temperature
 *   `T_w`, independent of its incoming direction. The outgoing normal
 *   speed follows a half-range Maxwellian (Rayleigh-shaped) distribution
 *   and the outgoing direction follows Lambert's cosine law.
 *
 * J. C. Maxwell (1879) proposed blending the two limits with a single
 * *accommodation coefficient* `sigma`: a fraction `sigma` of molecules
 * reflect diffusely and `(1 - sigma)` reflect specularly. This class
 * implements that blend, generalized to separate accommodation
 * coefficients for momentum (which branch is taken, `_momentum_acc`) and
 * for translational/rotational/vibrational energy (how far each internal
 * mode relaxes toward the wall temperature once the diffuse branch is
 * chosen, `_trans_acc` / `_rot_acc` / `_vib_acc`). Internal (rotational,
 * vibrational) energy exchange uses the Larsen-Borgnakke statistical
 * relaxation model, the same model DSMC uses for inelastic collisions
 * between molecules, applied here at the wall instead of between two
 * gas particles.
 *
 * ### Operating principle
 * 1. `operator()` draws four independent uniform samples from a hash of
 *    the incident/normal vectors (`atlas::sample_hashed_unit_interval`,
 *    so the interaction is stateless/reproducible without an RNG stream)
 *    and forwards them to `sample()`.
 * 2. `sample()` first decides the reflection branch against
 *    `_momentum_acc`. On the specular branch it mirrors the incident
 *    velocity about `normal`. On the diffuse branch it draws the outgoing
 *    velocity in the wall's local frame `(normal, tangent1, tangent2)`:
 *      - normal-direction speed `v_perp = v_mp * sqrt(-ln(U1))`
 *        (Rayleigh distribution, the flux-weighted half-Maxwellian);
 *      - in-plane speed `v_tangent = v_mp * sqrt(-ln(U2))` split into
 *        `(v_tangent * sin(theta), v_tangent * cos(theta))` via a
 *        Box-Muller-style polar transform, giving the correct 2D Gaussian
 *        tangential distribution;
 *    where `v_mp = sqrt(2 k_B T_w / m)` is the most probable molecular
 *    speed at the wall (`most_probable_speed()`). The tangent basis is
 *    built from the component of the incident velocity orthogonal to
 *    `normal`, or from a hashed seed direction if the incidence is
 *    exactly normal (degenerate tangent).
 * 3. `internal_energy()` re-samples rotational/vibrational energy on the
 *    diffuse branch only, blending the incident value with a
 *    wall-equilibrium draw by the relevant accommodation coefficient
 *    (`sample_internal_energy_mode`, using the same Rayleigh-in-2D
 *    construction as the translational speed, scaled by `k_B T_w`).
 *    The material-aware overload instead draws each mode directly from
 *    its equilibrium distribution given the molecule's rotational/
 *    vibrational degrees of freedom (dof) from `MaterialProperties`:
 *      - `MaxwellianInternalEnergyStyle::discrete` with `dof == 2` models
 *        a simple harmonic oscillator: the vibrational (or rigid-rotor)
 *        quantum level is drawn from the geometric/Boltzmann distribution
 *        `level = floor(-ln(U) * T_w / theta)`, `theta` a characteristic
 *        (rotational/vibrational) temperature, matching Bird's discrete
 *        internal-energy sampling for a two-dof rigid mode.
 *      - `MaxwellianInternalEnergyStyle::smooth` (or `discrete` with
 *        `dof != 2`) models a continuum of `dof` degrees of freedom via
 *        `sample_diffuse_smooth_energy`: `dof == 2` reduces to the closed
 *        form exponential draw `-ln(U) k_B T_w`; `dof > 2` uses Bird's
 *        acceptance-rejection scheme on the normalized energy
 *        `erm = E_r / (k_B T_w)` with acceptance probability
 *        `(erm / a)^a * exp(a - erm)`, `a = dof/2 - 1`, which reproduces
 *        the Boltzmann-weighted density of states for `dof` continuous
 *        modes.
 *      - `MaxwellianInternalEnergyStyle::none` leaves the mode
 *        unexcited (energy stays at zero / is not modeled).
 *
 * ### Derivations
 * Each derivation below is referenced by name (`D1`-`D4`) from the method
 * that uses it, so the sampling code can stay terse while still being
 * traceable back to the underlying probability density.
 *
 * **D1 — normal-speed Rayleigh draw (`v_perp`).**
 * A molecule effusing through/leaving a plane at normal speed `v_n > 0`
 * from an equilibrium gas is *flux-weighted*: a layer of gas at speed
 * `v_n` contributes particles to the surface at a rate proportional to
 * `v_n` itself (faster molecules arrive/leave more often per unit time),
 * on top of the ordinary Maxwell-Boltzmann population `exp(-v_n^2/v_mp^2)`.
 * So the normal-speed density is
 * `f(v_n) = (2 v_n / v_mp^2) * exp(-v_n^2 / v_mp^2)`, `v_n >= 0`
 * (normalized: `integral_0^inf f = 1`) — a Rayleigh distribution with
 * scale `v_mp / sqrt(2)`. Its CDF integrates in closed form:
 * `F(v_n) = 1 - exp(-v_n^2 / v_mp^2)`. Inverse-CDF (Smirnov) sampling
 * draws `U ~ Uniform(0, 1)` and solves `U = 1 - F(v_n)` (valid since
 * `1 - F(v_n)` is itself uniform on `(0, 1)` when `v_n` has CDF `F`):
 * `U = exp(-v_n^2 / v_mp^2)  =>  v_n = v_mp * sqrt(-ln U)` — exactly the
 * `vperp` / `v_perp` expression in `sample()`.
 *
 * **D2 — tangential-velocity Box-Muller draw (`v_tan1`, `v_tan2`).**
 * Unlike the normal component, the tangential velocity does not gate
 * whether a molecule reaches the wall, so it keeps the ordinary
 * (non-flux-weighted) isotropic 2D Gaussian of an equilibrium gas:
 * `f(v_x, v_y) = (1 / (pi v_mp^2)) * exp(-(v_x^2 + v_y^2) / v_mp^2)`,
 * matching single-axis variance `k_B T_w / m = v_mp^2 / 2`. Changing to
 * polar coordinates `(r, theta)` with `v_x = r sin(theta)`,
 * `v_y = r cos(theta)` and Jacobian `r`:
 * `f(r, theta) dr dtheta = (1/(pi v_mp^2)) exp(-r^2/v_mp^2) * r dr dtheta`,
 * which factors into an independent uniform angle `theta ~ Uniform(0,
 * 2*pi)` and a radial density `p(r) = (2r/v_mp^2) exp(-r^2/v_mp^2)` — the
 * *same* Rayleigh form as D1. Applying D1's inverse-CDF to `r` gives
 * `r = v_mp * sqrt(-ln U2)`, and drawing `theta = 2*pi*U3` then
 * `(v_x, v_y) = (r sin(theta), r cos(theta))` reproduces the 2D Gaussian
 * exactly — this is the Box-Muller transform, specialized to the case
 * where both output Gaussians share the tangent plane. `sample()` uses
 * this for `(vtan1, vtan2)` with `r = vtangent`.
 *
 * **D3 — continuous internal-energy rejection sampling (`dof > 2`).**
 * A classical system with `dof` continuous quadratic internal degrees of
 * freedom (e.g. `dof/2` independent harmonic-oscillator-like coordinate
 * pairs) has equilibrium energy density proportional to the density of
 * states times the Boltzmann factor: `p(E_r) ~ E_r^(dof/2 - 1) *
 * exp(-E_r / k_B T_w)` for `E_r >= 0` — a Gamma distribution with shape
 * `dof/2`. Writing `erm = E_r / (k_B T_w)` and `a = dof/2 - 1`:
 * `p(erm) ~ erm^a * exp(-erm)`. This peaks at `erm = a` (set
 * `d/derm[a ln(erm) - erm] = a/erm - 1 = 0`) with peak value
 * `a^a exp(-a)`; dividing by the peak gives the acceptance probability
 * `g(erm) = (erm/a)^a * exp(a - erm) <= 1` used in
 * `sample_diffuse_smooth_energy`. The method draws a candidate `erm`
 * uniformly on a bounded interval (`erm = 10 * energy_sample`, wide
 * enough that the Gamma tail beyond it is negligible for the `dof` this
 * class expects) and accepts it with probability `g(erm)` — standard
 * acceptance-rejection sampling against a (locally) uniform proposal,
 * following G. A. Bird's algorithm for polyatomic internal-energy
 * sampling (see References). `dof == 2` (`a == 0`) skips rejection
 * entirely: `g(erm) = exp(-erm) = 1` identically, so `p(erm) = exp(-erm)`
 * is already normalized and the direct inverse-CDF draw
 * `erm = -ln(sample)` applies (same construction as D1 without the
 * flux-weighting factor of `v_n`, since here the "radius" itself, not a
 * derived speed, is exponentially distributed in `erm`).
 *
 * **D4 — discrete quantum-level draw (`MaxwellianInternalEnergyStyle::discrete`).**
 * A non-degenerate harmonic-oscillator/rigid-rotor ladder with
 * characteristic temperature `theta` has level populations
 * `P(n) = (1 - exp(-theta/T_w)) * exp(-n*theta/T_w)`, `n = 0, 1, 2, ...`
 * (a geometric distribution; the prefactor normalizes the geometric
 * series to sum to 1). This is recovered from a *continuous* exponential
 * variate by flooring: let `X = -ln(U) * (T_w/theta)`, the inverse-CDF
 * draw of an `Exponential(rate = theta/T_w)` variable (solve
 * `U = exp(-X * theta/T_w)`); then `n = floor(X)` has probability mass
 * `Pr[n <= X < n+1] = exp(-n*theta/T_w) - exp(-(n+1)*theta/T_w) =
 * exp(-n*theta/T_w) * (1 - exp(-theta/T_w))`, exactly `P(n)` above. This
 * is `level = static_cast<int>(-log(sample) * _temperature / quantum)` in
 * `sample_diffuse_rotational_energy` / `sample_diffuse_vibrational_energy`;
 * the returned energy `level * k_B * theta` is the level index times the
 * oscillator quantum spacing.
 *
 * ### References
 * - J. C. Maxwell, "On stresses in rarefied gases arising from
 *   inequalities of temperature," Philosophical Transactions of the
 *   Royal Society of London, 170, 1879, pp. 231-256. (origin of the
 *   accommodation-coefficient specular/diffuse blend)
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994. (canonical DSMC reference;
 *   diffuse-wall velocity sampling and the internal-energy
 *   acceptance-rejection / discrete-level sampling algorithms implemented
 *   here follow this text)
 * - C. Borgnakke and P. S. Larsen, "Statistical collision model for
 *   Monte Carlo simulation of polyatomic gas mixture," Journal of
 *   Computational Physics, 18(4), 1975, pp. 405-420. (Larsen-Borgnakke
 *   internal-energy relaxation model reused here for wall accommodation)
 * - C. Cercignani and M. Lampis, "Kinetic models for gas-surface
 *   interactions," Transport Theory and Statistical Physics, 1(2), 1971,
 *   pp. 101-114. (the more general CLL scattering kernel; this class
 *   implements the simpler Maxwell specular/diffuse blend rather than
 *   full CLL tangential-momentum scattering)
 */

namespace atlas {

/**
 * @brief Selects how a diffusely-reflected molecule's rotational or
 *        vibrational energy is redrawn at the wall.
 *
 * Used independently for the rotational (`_rot_style`) and vibrational
 * (`_vib_style`) modes; see the "Operating principle" section of this
 * file's documentation for the sampling procedure each value selects.
 */
enum struct MaxwellianInternalEnergyStyle : int {
    /** Mode is not excited; its energy is not modeled (stays zero). */
    none,
    /** Continuous-dof Boltzmann distribution (closed form for dof == 2,
     *  Bird's acceptance-rejection scheme for dof > 2). */
    smooth,
    /** Quantized simple-harmonic-oscillator / rigid-rotor levels
     *  (dof == 2 only; falls back to `smooth` otherwise). */
    discrete
};

namespace detail {

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    maxwellian_unit_sample(const float u) noexcept {
        // Compare `eps` by value; std::max would bind the namespace-scope
        // constexpr to a `const float&` (odr-use), leaving it undefined in
        // device code.
        return u > atlas::eps ? u : atlas::eps;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    maxwellian_tangent_seed(const Vector3& normal, const Vector3& seed) noexcept {
        return atlas::orthogonal_unit_vector(normal, seed, atlas::tol);
    }

}

/**
 * @brief Maxwell-model gas-surface interaction: blends specular and
 *        fully-diffuse (equilibrium-Maxwellian) molecular reflection at a
 *        wall of temperature `_temperature`.
 *
 * A stateless, device-callable value type: every query method is a pure
 * function of its arguments plus the accommodation parameters below (no
 * RNG stream is carried — randomness comes from hashing the incident
 * state, see `atlas::sample_hashed_unit_interval`). See this file's
 * top-of-file documentation for the physical model and the sampling
 * derivation.
 *
 * Construct via `Builder`; the four accommodation coefficients
 * (`_momentum_acc`, `_trans_acc`, `_rot_acc`, `_vib_acc`) each lie in
 * `[0, 1]`, where `1` is fully diffuse/fully thermalized to `_temperature`
 * and `0` is fully specular/energy-preserving.
 */
class MaxwellianSurfaceInteraction final {
public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_HOST void
    set_temperature(float temperature) noexcept;

    ATLAS_HOST void
    set_molecular_mass(float molecular_mass) noexcept;

    ATLAS_HOST void
    set_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST void
    set_trans_acc(float trans_acc) noexcept;

    ATLAS_HOST void
    set_rot_acc(float rot_acc) noexcept;

    ATLAS_HOST void
    set_vib_acc(float vib_acc) noexcept;

    ATLAS_HOST void
    set_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST void
    set_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST void
    set_accommodation(float momentum_acc, float trans_acc, float rot_acc, float vib_acc) noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    temperature() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    molecular_mass() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    momentum_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    trans_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    rot_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    vib_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD MaxwellianInternalEnergyStyle
    rot_style() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD MaxwellianInternalEnergyStyle
    vib_style() const noexcept;

    /**
     * @brief Most probable speed `v_mp = sqrt(2 k_B T / m)` of the
     *        equilibrium Maxwell-Boltzmann speed distribution at the wall.
     *
     * This is the natural velocity scale of the half-range Maxwellian used
     * by `sample()`: the sampled normal and tangential speed components
     * are each `v_mp` times a dimensionless random factor.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    most_probable_speed() const noexcept {
        return atlas::sqrt_nonnegative(
            2.0f * atlas::boltzmann_constant * _temperature / _molecular_mass);
    }

    /**
     * @brief Reflects `incident` off a wall with normal `normal`,
     *        stochastically choosing the specular or diffuse branch.
     *
     * Derives all random numbers `sample()` needs from hashes of
     * `incident`/`normal` (via `atlas::sample_hashed_unit_interval` with
     * distinct salts per draw), so the interaction is a pure, stateless
     * function of the incident state — no RNG stream needs to be threaded
     * through the caller. See `sample()` for the branch/velocity
     * derivation and this file's top-of-file documentation for the
     * underlying physical model.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    operator()(const Vector3& incident, const Vector3& normal) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const float perpendicular_sample = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float theta_sample = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        const float tangent_sample = atlas::sample_hashed_unit_interval(
            incident + normal * 3.0f,
            4.19f);

        const Vector3 tangent_seed(
            atlas::sample_hashed_unit_interval(incident, 5.11f),
            atlas::sample_hashed_unit_interval(normal, 6.23f),
            atlas::sample_hashed_unit_interval(incident + normal, 7.37f));

        return sample(
            incident,
            normal,
            branch_sample,
            perpendicular_sample,
            theta_sample,
            tangent_sample,
            tangent_seed);
    }

    /**
     * @brief Deterministic core of the interaction, given pre-drawn
     *        uniform samples in `[0, 1)`.
     *
     * `branch_sample > _momentum_acc` takes the specular branch
     * (`atlas::reflected`: mirror `incident` about `normal`, energy- and
     * angle-preserving). Otherwise the diffuse branch draws a velocity in
     * the local wall frame `(normal, tangent1, tangent2)`:
     *   - normal speed `v_perp = v_mp * sqrt(-ln(perpendicular_sample))`
     *     — the flux-weighted half-Maxwellian Rayleigh draw, see this
     *     file's Derivation D1;
     *   - tangential speed `v_tangent = v_mp * sqrt(-ln(tangent_sample))`
     *     split into `(sin(theta), cos(theta)) * v_tangent` via
     *     `theta = 2*pi*theta_sample` — the Box-Muller polar transform of
     *     the equilibrium tangential Gaussian, see Derivation D2.
     * `tangent1` is the incident velocity's component orthogonal to
     * `normal` (grazing incidence reuses the caller-supplied
     * `tangent_seed` to pick an arbitrary tangent when that projection is
     * zero); `tangent2` completes a right-handed basis via `normal x
     * tangent1`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    sample(const Vector3& incident,
           const Vector3& normal,
           const float branch_sample,
           const float perpendicular_sample,
           const float theta_sample,
           const float tangent_sample,
           const Vector3& tangent_seed) const noexcept {
        if (branch_sample > _momentum_acc) {
            return atlas::reflected(incident, normal);
        }

        const float vrm      = most_probable_speed();
        const float vperp    = vrm * atlas::sqrt_nonnegative(-std::log(detail::maxwellian_unit_sample(perpendicular_sample)));
        const float theta    = 2.0f * atlas::pi * theta_sample;
        const float vtangent = vrm * atlas::sqrt_nonnegative(-std::log(detail::maxwellian_unit_sample(tangent_sample)));
        const float vtan1    = vtangent * std::sin(theta);
        const float vtan2    = vtangent * std::cos(theta);

        Vector3 tangent1 = atlas::reject(incident, normal);

        if (tangent1.length_squared() == 0.0f) {
            tangent1 = detail::maxwellian_tangent_seed(normal, tangent_seed);
        } else {
            tangent1 = tangent1.normalized();
        }

        const Vector3 tangent2 = atlas::cross(normal, tangent1);

        return normal * vperp + tangent1 * vtan1 + tangent2 * vtan2;
    }

    /**
     * @brief Redraws all three internal-energy modes (translational,
     *        rotational, vibrational) toward wall equilibrium, blended by
     *        each mode's own accommodation coefficient.
     *
     * Always applies the diffuse-branch relaxation (`sample_internal_energy`)
     * unconditionally — use the `(incident_energy, incident_velocity,
     * normal[, material])` overloads when the specular/diffuse branch
     * decision (`_momentum_acc`) should gate whether internal energy is
     * touched at all, matching the translational reflection branch in
     * `sample()`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident) const noexcept {
        const Vector3 seed(incident.translational, incident.rotational, incident.vibrational);

        return sample_internal_energy(
            incident,
            atlas::sample_hashed_unit_interval(seed, 8.11f),
            atlas::sample_hashed_unit_interval(seed, 9.23f),
            atlas::sample_hashed_unit_interval(seed, 10.37f),
            atlas::sample_hashed_unit_interval(seed, 11.41f),
            atlas::sample_hashed_unit_interval(seed, 12.53f),
            atlas::sample_hashed_unit_interval(seed, 13.67f));
    }

    /**
     * @brief Redraws internal energy only on the same diffuse branch that
     *        `sample()` would take for this incident/normal pair (uses
     *        the identical `RANDOM_HASH_SALT_MIX` hash, so the
     *        translational and internal-energy branch decisions agree for
     *        one interaction event). Leaves `incident_energy` unchanged on
     *        the specular branch.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Vector3& incident_velocity,
                    const Vector3& normal) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident_velocity + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        return branch_sample > _momentum_acc ? incident_energy : internal_energy(incident_energy);
    }

    /**
     * @brief Material-aware variant: on the diffuse branch, redraws
     *        rotational/vibrational energy from their equilibrium
     *        distributions for the molecule's actual degrees of freedom
     *        (`MaterialProperties::rotational_dof` /
     *        `vibrational_dof`) via `sample_diffuse_rotational_energy` /
     *        `sample_diffuse_vibrational_energy`, rather than blending the
     *        incident value by a flat accommodation coefficient. The
     *        translational mode is left untouched (specular translational
     *        energy is assumed unaffected by internal-mode relaxation
     *        here).
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Vector3& incident_velocity,
                    const Vector3& normal,
                    const MaterialProperties& material) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident_velocity + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        if (branch_sample > _momentum_acc) {
            return incident_energy;
        }

        const Vector3 seed(
            incident_energy.translational + incident_velocity.x,
            incident_energy.rotational + incident_velocity.y,
            incident_energy.vibrational + incident_velocity.z);

        return FluidInternalEnergy {
            incident_energy.translational,
            sample_diffuse_rotational_energy(material, seed),
            sample_diffuse_vibrational_energy(material, seed)
        };
    }

    /**
     * @brief Blends each of the three internal-energy modes toward wall
     *        equilibrium independently via `sample_internal_energy_mode`,
     *        using each mode's own accommodation coefficient
     *        (`_trans_acc`, `_rot_acc`, `_vib_acc`) and an independent
     *        pair of hashed uniforms per mode.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy
    sample_internal_energy(const FluidInternalEnergy& incident,
                           const float trans_sample,
                           const float trans_theta_sample,
                           const float rot_sample,
                           const float rot_theta_sample,
                           const float vib_sample,
                           const float vib_theta_sample) const noexcept {
        return FluidInternalEnergy {
            sample_internal_energy_mode(incident.translational, _trans_acc, trans_sample, trans_theta_sample),
            sample_internal_energy_mode(incident.rotational, _rot_acc, rot_sample, rot_theta_sample),
            sample_internal_energy_mode(incident.vibrational, _vib_acc, vib_sample, vib_theta_sample)
        };
    }

    /**
     * @brief Partial-accommodation draw for a single internal-energy
     *        mode: interpolates between exactly preserving `incident` (at
     *        `acc == 0`) and a full wall-equilibrium draw (at
     *        `acc == 1`) for intermediate accommodation coefficients.
     *
     * Represents the retained and thermal contributions as two segments
     * in an abstract 2D "energy-amplitude" space and combines them by the
     * law of cosines at a random relative phase, then squares the
     * resultant length to get an energy:
     *   - `magnitude = sqrt(incident * (1 - acc) / (k_B T_w))` — the
     *     amplitude carried over from the incident state, shrinking as
     *     `acc -> 1`;
     *   - `radius = sqrt(-acc * ln(sample))` — a Rayleigh-distributed
     *     thermal amplitude scaled by `acc`, vanishing as `acc -> 0`;
     *   - `phase = cos(2*pi*theta_sample)` — a uniform random relative
     *     angle between the two contributions;
     *   - result `= k_B T_w * (radius^2 + magnitude^2 +
     *     2*radius*magnitude*phase)`.
     *
     * At `acc == 1` this reduces exactly to the closed-form 2-dof
     * equilibrium draw `-ln(sample) * k_B T_w` (see
     * `sample_diffuse_smooth_energy`'s `dof == 2` case); at `acc == 0` it
     * reduces exactly to `incident`. This is the classic law-of-cosines
     * construction used for incomplete internal-energy accommodation in
     * gas-surface scattering models that extend Maxwell/CLL-style wall
     * interactions with partial (rather than all-or-nothing) thermal
     * relaxation (cf. R. G. Lord's extension of the Cercignani-Lampis-Lord
     * kernel to internal energy, Phys. Fluids A 3(4), 1991, pp. 706-710).
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    sample_internal_energy_mode(const float incident,
                                const float acc,
                                const float sample,
                                const float theta_sample) const noexcept {
        const float wall_energy = atlas::boltzmann_constant * _temperature;
        const float safe_wall   = wall_energy > 0.0f ? wall_energy : std::numeric_limits<float>::min();
        const float safe_energy = std::max(incident, 0.0f);
        const float magnitude   = atlas::sqrt_nonnegative(safe_energy * (1.0f - acc) / safe_wall);
        const float radius      = atlas::sqrt_nonnegative(-acc * std::log(detail::maxwellian_unit_sample(sample)));
        const float phase       = std::cos(2.0f * atlas::pi * theta_sample);

        return safe_wall * (radius * radius + magnitude * magnitude + 2.0f * radius * magnitude * phase);
    }

    /**
     * @brief Draws a full-equilibrium rotational energy for a molecule
     *        with `material.rotational_dof` rotational degrees of
     *        freedom, per `_rot_style`.
     *
     * Returns `0` if rotation is unmodeled (`MaxwellianInternalEnergyStyle::none`)
     * or the molecule has fewer than 2 rotational dof (e.g. a monatomic
     * species). Uses `material.rotational_temperature` as the
     * characteristic temperature for the discrete (quantized rigid-rotor,
     * `dof == 2`) branch if the material supplies one, otherwise falls
     * back to the wall temperature `_temperature` — see this file's
     * Derivation D4 for why flooring a scaled exponential draw reproduces
     * the correct geometric level distribution. The `dof != 2` /
     * `smooth`-style path delegates to `sample_diffuse_smooth_energy`
     * (Derivation D3).
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    sample_diffuse_rotational_energy(const MaterialProperties& material,
                                     const Vector3& seed) const noexcept {
        const int dof = material.rotational_dof.has_value() ? *material.rotational_dof : 0;
        if (_rot_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
            return 0.0f;
        }

        if (_rot_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
            const float rot_temperature = material.rotational_temperature.has_value()
                ? *material.rotational_temperature
                : _temperature;
            const float quantum         = rot_temperature > atlas::eps ? rot_temperature : atlas::eps;
            const float sample          = detail::maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, 14.11f));
            const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
            return static_cast<float>(level) * atlas::boltzmann_constant * quantum;
        }

        return sample_diffuse_smooth_energy(dof, seed, 15.23f);
    }

    /**
     * @brief Draws a full-equilibrium vibrational energy for a molecule
     *        with `material.vibrational_dof` vibrational degrees of
     *        freedom, per `_vib_style`. Mirrors
     *        `sample_diffuse_rotational_energy`, using
     *        `material.characteristic_vibrational_temperature` (falling
     *        back to `_temperature`) as the discrete-branch quantum scale
     *        — i.e. a simple-harmonic-oscillator model of the vibrational
     *        mode when `_vib_style == discrete` and `dof == 2`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    sample_diffuse_vibrational_energy(const MaterialProperties& material,
                                      const Vector3& seed) const noexcept {
        const int dof = material.vibrational_dof.has_value() ? *material.vibrational_dof : 0;
        if (_vib_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
            return 0.0f;
        }

        if (_vib_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
            const float vib_temperature = material.characteristic_vibrational_temperature.has_value()
                ? *material.characteristic_vibrational_temperature
                : _temperature;
            const float quantum         = vib_temperature > atlas::eps ? vib_temperature : atlas::eps;
            const float sample          = detail::maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, 16.37f));
            const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
            return static_cast<float>(level) * atlas::boltzmann_constant * quantum;
        }

        return sample_diffuse_smooth_energy(dof, seed, 17.41f);
    }

    /**
     * @brief Draws an equilibrium internal energy for a continuum of
     *        `dof` degrees of freedom at temperature `_temperature`.
     *
     * `dof == 2` (e.g. a single rotational mode of a linear/diatomic
     * molecule) has the closed-form exponential density
     * `p(E) ~ exp(-E / k_B T)`, sampled directly as
     * `-ln(sample) * k_B T` (inverse-CDF sampling).
     *
     * `dof > 2` has no closed-form inverse CDF, so this uses G. A. Bird's
     * acceptance-rejection scheme — the full derivation of the target
     * density, the peak-normalized acceptance probability, and why
     * `dof == 2` needs no rejection at all is this file's Derivation D3.
     * Loops until a candidate is accepted (each iteration advances the
     * hash salt by `0.37` so retries are independent draws from the same
     * deterministic hash seed).
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    sample_diffuse_smooth_energy(const int dof,
                                 const Vector3& seed,
                                 const float salt) const noexcept {
        if (dof == 2) {
            const float sample = detail::maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, salt));
            return -std::log(sample) * atlas::boltzmann_constant * _temperature;
        }

        const float a = 0.5f * static_cast<float>(dof) - 1.0f;
        if (!(a > 0.0f)) {
            return 0.0f;
        }

        for (int i = 0;; ++i) {
            const float energy_sample = atlas::sample_hashed_unit_interval(seed, salt + static_cast<float>(i) * 0.37f);
            const float accept_sample = atlas::sample_hashed_unit_interval(seed, salt + static_cast<float>(i) * 0.37f + 0.19f);
            const float erm           = 10.0f * energy_sample;
            const float b             = std::pow(erm / a, a) * std::exp(a - erm);
            if (b > accept_sample) {
                return erm * atlas::boltzmann_constant * _temperature;
            }
        }

        return 0.0f;
    }

private:
    float _temperature { 273.15f };
    float _molecular_mass { 1.0f };
    float _momentum_acc { 1.0f };
    float _trans_acc { 1.0f };
    float _rot_acc { 1.0f };
    float _vib_acc { 1.0f };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

/**
 * @brief Fluent builder for `MaxwellianSurfaceInteraction`; defaults match
 *        the class's in-class defaults (`T = 273.15 K`, unit molecular
 *        mass, fully diffuse/fully accommodating on every mode, smooth
 *        internal-energy sampling). `validate()` runs in `build()` /
 *        `make_host_shared()`.
 */
class MaxwellianSurfaceInteraction::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_HOST Builder&
    with_molecular_mass(float molecular_mass) noexcept;

    ATLAS_HOST Builder&
    with_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST Builder&
    with_trans_acc(float trans_acc) noexcept;

    ATLAS_HOST Builder&
    with_rot_acc(float rot_acc) noexcept;

    ATLAS_HOST Builder&
    with_vib_acc(float vib_acc) noexcept;

    ATLAS_HOST Builder&
    with_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST Builder&
    with_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST Builder&
    with_accommodation(float momentum_acc, float trans_acc, float rot_acc, float vib_acc) noexcept;

    ATLAS_HOST ATLAS_NODISCARD MaxwellianSurfaceInteraction
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<MaxwellianSurfaceInteraction>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    float _temperature { 273.15f };
    float _molecular_mass { 1.0f };
    float _momentum_acc { 1.0f };
    float _trans_acc { 1.0f };
    float _rot_acc { 1.0f };
    float _vib_acc { 1.0f };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

using MaxwellianSurfaceInteractionHostPtr = atlas::host_shared_ptr<MaxwellianSurfaceInteraction>;

using MaxwellianSurfaceInteractionDevicePtr = atlas::device_shared_ptr<MaxwellianSurfaceInteraction>;

}
