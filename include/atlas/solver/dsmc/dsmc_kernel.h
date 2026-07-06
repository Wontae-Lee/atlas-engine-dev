#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/dsmc/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <cstddef>
#include <type_traits>

/**
 * @file dsmc_kernel.h
 * @brief Runtime-selectable dispatcher over the three DSMC binary
 *        collision models (HS/VHS/VSS), plus `sigma_g`: the collision
 *        rate factor consumed by the no-time-counter (NTC) collision
 *        selection scheme.
 *
 * @details
 * ### Background — why `sigma * g`, not just `sigma`
 * DSMC does not test every particle pair in a cell for collision (that
 * would be `O(n^2)`); instead it estimates how many collisions *should*
 * happen in a cell this timestep and randomly selects that many
 * candidate pairs to actually collide (Bird's no-time-counter method).
 * The expected collision rate for a pair of species is proportional to
 * `sigma(g) * g` (cross-section times relative speed) — not `sigma`
 * alone — because a pair with a larger cross-section *and* a pair that
 * is closing faster are both more likely to actually collide per unit
 * time; this product is the quantity NTC needs both to size its
 * candidate pool (via `max(sigma * g)` over recent pairs) and to accept
 * or reject each candidate pair (accept with probability
 * `(sigma * g) / max(sigma * g)`). `sigma_g()` below is exactly this
 * per-pair product, evaluated through whichever collision model
 * `DsmcKernel` currently holds.
 *
 * ### Operating principle
 * Follows the same tagged-union `DeviceVariant` pattern as
 * `SurfaceInteractionKernel`/`PostColliderKernel` (see those files for
 * why): a `DsmcKernelType` tag selects which of `HardSphereKernel`/
 * `VariableHardSphereKernel`/`VariableSoftSphereKernel` is the active
 * union payload. `cross_section()` and `operator()` (the collision
 * outcome) dispatch to the active payload's own implementation — see
 * `hard_sphere_kernel.h`, `variable_hard_sphere_kernel.h`,
 * `variable_soft_sphere_kernel.h` for what each model actually computes.
 * `pair_parameters()` precomputes the pairwise-averaged material
 * constants (`reduced_mass`, `viscosity_index`, `scattering_parameter`,
 * `reference_diameter`/`reference_temperature`) shared by the VHS/VSS
 * formulas, so callers that need them outside a collision kernel call
 * (e.g. for diagnostics) don't have to duplicate the mixing rule.
 */

namespace atlas {

/**
 * @brief Selects which DSMC binary collision model a `DsmcKernel`
 *        applies; see `hard_sphere_kernel.h`, `variable_hard_sphere_kernel.h`,
 *        `variable_soft_sphere_kernel.h` for the physics of each.
 */
enum struct DsmcKernelType : int {
    /** Constant cross-section, isotropic scattering. */
    hard_sphere,
    /** Speed-dependent cross-section (correct viscosity-temperature
     *  exponent), isotropic scattering. */
    variable_hard_sphere,
    /** Same cross-section as `variable_hard_sphere`, anisotropic
     *  (forward-peaked-tunable) scattering. */
    variable_soft_sphere
};

/**
 * @brief Pairwise-averaged material constants shared by the VHS/VSS
 *        cross-section and scattering formulas (see
 *        `DsmcKernel::pair_parameters`); `valid` reports whether both
 *        species had a positive combined mass (i.e. whether the other
 *        fields are meaningful).
 */
struct DsmcPairParameters final {
    float reference_diameter {};
    float reference_temperature {};
    float viscosity_index { 0.5f };
    float scattering_parameter { 1.0f };
    float reduced_mass {};
    bool valid {};
};

/**
 * @brief Tagged-union wrapper letting a `DsmcSolver` collide particle
 *        pairs through whichever `DsmcKernelType` it was configured
 *        with, without virtual dispatch. See this file's top-of-file
 *        documentation for the `DeviceVariant` pattern and why
 *        `sigma_g()` (not `cross_section()` alone) drives NTC selection.
 */
struct DsmcKernel final {

    DsmcKernelType type = DsmcKernelType::hard_sphere;

    union {

        HardSphereKernel hard_sphere;

        VariableHardSphereKernel variable_hard_sphere;

        VariableSoftSphereKernel variable_soft_sphere;
    };

    ATLAS_ALL_DEVICE
    DsmcKernel() noexcept;

    ATLAS_ALL_DEVICE explicit DsmcKernel(DsmcKernelType type) noexcept;

    ATLAS_ALL_DEVICE
    DsmcKernel(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE ~DsmcKernel() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, DsmcKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit DsmcKernel(const Payload& op);

    /** @brief Static dispatch to `type`'s `cross_section()`; does not
     *  require a constructed `DsmcKernel` instance. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(DsmcKernelType type,
                  const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    /** @brief Pairwise-averaged VHS/VSS material constants for `lhs`/
     *  `rhs`; see `DsmcPairParameters`. Model-agnostic (does not depend
     *  on the active `type`). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static DsmcPairParameters
    pair_parameters(const MaterialProperties& lhs,
                    const MaterialProperties& rhs) noexcept;

    /** @brief Dispatches to the active payload's collision outcome
     *  (elastic scattering; see the HS/VHS/VSS files for each model's
     *  scattering law). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const MaterialProperties& lhs,
               const MaterialProperties& rhs) const noexcept;

    /**
     * @brief `sigma(g) * g` for species `species_i`/`species_j` at
     *        squared relative speed `relative_speed_squared`, the NTC
     *        collision-rate factor — see this file's top-of-file
     *        documentation for why the product (not `sigma` alone)
     *        drives DSMC pair selection. Returns `0` if
     *        `relative_speed_squared <= 0` (stationary pair, no
     *        collision rate).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sigma_g(const MaterialProperties* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            float relative_speed_squared) const noexcept;
};

namespace detail {

    using DsmcKernelVariant = DeviceVariant<
        DsmcKernel,
        DsmcKernelType,
        DsmcKernelType::hard_sphere,
        DeviceVariantCase<DsmcKernelType::hard_sphere, &DsmcKernel::hard_sphere>,
        DeviceVariantCase<DsmcKernelType::variable_hard_sphere, &DsmcKernel::variable_hard_sphere>,
        DeviceVariantCase<DsmcKernelType::variable_soft_sphere, &DsmcKernel::variable_soft_sphere>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
    struct DsmcCrossSection {
        const MaterialProperties& lhs;
        const MaterialProperties& rhs;
        float relative_speed;
        template <typename Tag>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
        operator()(Tag) const noexcept {
            using Kernel = typename Tag::type;
            return Kernel::cross_section(lhs, rhs, relative_speed);
        }
    };
    struct DsmcCollide {
        Float3& lhs_velocity;
        Float3& rhs_velocity;
        const MaterialProperties& lhs;
        const MaterialProperties& rhs;
        template <typename K>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        operator()(const K& kernel) const noexcept { kernel(lhs_velocity, rhs_velocity, lhs, rhs); }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel() noexcept {
    detail::DsmcKernelVariant::construct(*this, DsmcKernelType::hard_sphere);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const DsmcKernelType type) noexcept {
    detail::DsmcKernelVariant::construct(*this, type);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, DsmcKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const Payload& op) {
    detail::DsmcKernelVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcPairParameters
DsmcKernel::pair_parameters(const MaterialProperties& lhs,
                            const MaterialProperties& rhs) noexcept {
    DsmcPairParameters pair {};
    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)) {
        return pair;
    }

    pair.reduced_mass    = lhs_mass * rhs_mass / mass_sum;
    pair.viscosity_index = (lhs.viscosity_index.value_or(0.5f)
                            + rhs.viscosity_index.value_or(0.5f))
        * 0.5f;
    pair.scattering_parameter = (lhs.scattering_parameter.value_or(1.0f)
                                 + rhs.scattering_parameter.value_or(1.0f))
        * 0.5f;

    if (lhs.reference_diameter.has_value() && rhs.reference_diameter.has_value()) {
        pair.reference_diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * 0.5f;
    }
    if (lhs.reference_temperature.has_value() && rhs.reference_temperature.has_value()) {
        pair.reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * 0.5f;
    }

    pair.valid = pair.reduced_mass > 0.0f;
    return pair;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::cross_section(const DsmcKernelType type,
                          const MaterialProperties& lhs,
                          const MaterialProperties& rhs,
                          const float relative_speed) noexcept {
    return detail::DsmcKernelVariant::visit_type(
        type,
        detail::DsmcCrossSection { lhs, rhs, relative_speed },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcKernel::operator()(Float3& lhs_velocity,
                       Float3& rhs_velocity,
                       const MaterialProperties& lhs,
                       const MaterialProperties& rhs) const noexcept {
    detail::DsmcKernelVariant::apply(
        *this,
        detail::DsmcCollide { lhs_velocity, rhs_velocity, lhs, rhs });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::sigma_g(const MaterialProperties* properties_ptr,
                    const std::size_t species_i,
                    const std::size_t species_j,
                    const float relative_speed_squared) const noexcept {
    if (!(relative_speed_squared > 0.0f)) {
        return 0.0f;
    }

    const float relative_speed = atlas::sqrt_nonnegative(relative_speed_squared);
    return DsmcKernel::cross_section(
               type,
               properties_ptr[species_i],
               properties_ptr[species_j],
               relative_speed)
        * relative_speed;
}

}
