#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/dsmc/kernel/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_soft_sphere_kernel.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @brief Compile-time contract every DSMC collision kernel leaf must satisfy.
 *
 * A conforming leaf exposes a *static* @c cross_section returning the collision cross
 * section (m^2) for two materials at a relative speed, and a *const* call operator that
 * scatters the two velocities in place, drawing its variates from a caller-supplied
 * @ref atlas::default_random_engine. Modelling both as required members lets
 * @ref DsmcKernel dispatch to any leaf uniformly through @ref DsmcKernelVariant.
 *
 * The engine is passed by reference rather than owned by the leaf so the leaves stay
 * stateless PODs, which is what keeps @ref DsmcKernel trivially copyable into a device lambda.
 *
 * @tparam K The candidate kernel leaf type.
 */
template <typename K>
concept ConceptDsmcKernel = requires(const K kernel,
                                     Float3 velocity,
                                     const Material material,
                                     float relative_speed,
                                     default_random_engine& engine) {
    { K::cross_section(material, material, relative_speed) } -> std::same_as<float>;
    { kernel(velocity, velocity, material, material, engine) } -> std::same_as<void>;
};

static_assert(ConceptDsmcKernel<HardSphereKernel>);         ///< HS leaf honours the kernel contract.
static_assert(ConceptDsmcKernel<VariableHardSphereKernel>); ///< VHS leaf honours the kernel contract.
static_assert(ConceptDsmcKernel<VariableSoftSphereKernel>); ///< VSS leaf honours the kernel contract.

/**
 * @brief Trivially-copyable tagged union over the DSMC collision-kernel leaves.
 *
 * Wraps one of @ref HardSphereKernel, @ref VariableHardSphereKernel, or
 * @ref VariableSoftSphereKernel and dispatches to it via @ref DsmcKernelVariant. Because
 * the leaves are stateless PODs the whole object is trivially copyable, so a @ref DsmcSolver
 * captures it by value into a `__host__ __device__` collision lambda — that is why this is a
 * @ref DeviceVariant rather than a @ref HostVariant. @c type is the active-member
 * discriminator and is public so the variant machinery (and callers such as
 * @ref DsmcSolver::kernel_type) can read it directly.
 */
class DsmcKernel final {
public:
    DsmcKernelType type = DsmcKernelType::hard_sphere; ///< Discriminator naming the active union member.

    /** @brief Storage shared by the leaf kernels; exactly the member named by @ref type is live. */
    union {

        HardSphereKernel hard_sphere; ///< Active when @ref type is @ref DsmcKernelType::hard_sphere.

        VariableHardSphereKernel variable_hard_sphere; ///< Active for @ref DsmcKernelType::variable_hard_sphere.

        VariableSoftSphereKernel variable_soft_sphere; ///< Active for @ref DsmcKernelType::variable_soft_sphere.
    };

    /** @brief Default-constructs to the hard-sphere leaf. */
    ATLAS_ALL_DEVICE
    DsmcKernel() noexcept;

    /**
     * @brief Constructs with the leaf named by @p kernel_type activated.
     * @param kernel_type Which kernel leaf to make live.
     */
    ATLAS_ALL_DEVICE explicit DsmcKernel(DsmcKernelType kernel_type) noexcept;

    /** @brief Defaulted copy constructor; leaves are trivially copyable so a bitwise copy is valid. */
    ATLAS_ALL_DEVICE
    DsmcKernel(const DsmcKernel& other) noexcept = default;

    /** @brief Defaulted copy assignment; trivial for the POD leaves. */
    ATLAS_ALL_DEVICE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept = default;

    /** @brief Defaulted destructor; the leaves hold no resources. */
    ATLAS_ALL_DEVICE
    ~DsmcKernel() noexcept = default;

    /**
     * @brief Collision cross section for a material pair at a given relative speed.
     *
     * Dispatches to the active leaf's static @c cross_section.
     *
     * @param lhs           First collision partner's material.
     * @param rhs           Second collision partner's material.
     * @param relative_speed Magnitude of the relative velocity (m/s); some leaves ignore it.
     * @return Cross section in m^2, or `0` if the active leaf rejects the pair; `0` when
     *         no leaf matched the tag.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    cross_section(const Material& lhs, const Material& rhs, float relative_speed) const noexcept;

    /**
     * @brief Convenience product `sigma * g` used directly by the NTC acceptance test.
     *
     * Takes the *squared* relative speed to avoid a redundant square root at the call
     * site, computes `relative_speed = sqrt(...)`, and returns `cross_section * relative_speed`.
     * Looks the two materials up in @p materials by species index without a bounds check —
     * the System has already clamped every species id to the dictionary length.
     *
     * @param materials             Device material dictionary.
     * @param lhs_species           First partner's species index into @p materials.
     * @param rhs_species           Second partner's species index into @p materials.
     * @param relative_speed_squared Squared relative speed (m^2/s^2).
     * @return `sigma * g` in m^3/s, or `0` when the squared speed is non-positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sigma_g(const Material* materials,
            std::size_t lhs_species,
            std::size_t rhs_species,
            float relative_speed_squared) const noexcept;

    /**
     * @brief Scatters a collision pair in place through the active leaf.
     *
     * Dispatches to the active leaf's call operator, which conserves momentum and energy
     * while randomizing the post-collision relative direction. Both velocities are updated.
     *
     * @param lhs_velocity First partner's velocity (m/s); updated in place.
     * @param rhs_velocity Second partner's velocity (m/s); updated in place.
     * @param lhs          First partner's material (mass, scattering parameter).
     * @param rhs          Second partner's material.
     * @param engine       Generator supplying the scatter variates; advanced by the active leaf.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs,
               default_random_engine& engine) const noexcept;
};

/**
 * @brief Static dispatcher binding each @ref DsmcKernelType tag to its union member.
 *
 * Instantiates @ref DeviceVariant over @ref DsmcKernel with @ref DsmcKernelType::hard_sphere
 * as the fallback arm, so @ref DsmcKernel's members can construct/visit/apply the correct
 * leaf without a hand-written switch.
 */
using DsmcKernelVariant = DeviceVariant<
    DsmcKernel,
    DsmcKernelType,
    DsmcKernelType::hard_sphere,
    DeviceVariantCase<DsmcKernelType::hard_sphere, &DsmcKernel::hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_hard_sphere, &DsmcKernel::variable_hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_soft_sphere, &DsmcKernel::variable_soft_sphere>>;

/**
 * @brief Visitor that evaluates any leaf's static @c cross_section with captured arguments.
 *
 * Passed to @ref DsmcKernelVariant::visit so the same call site works for every leaf; the
 * template call operator receives the active leaf type (unused as a value, only for its type).
 */
class DsmcCrossSection {
public:
    const Material& lhs;  ///< First partner's material forwarded to the leaf.
    const Material& rhs;  ///< Second partner's material forwarded to the leaf.
    float relative_speed; ///< Relative speed (m/s) forwarded to the leaf.
    /**
     * @brief Invokes the leaf's static cross section.
     * @tparam K The active kernel leaf type (value ignored; only its type is used).
     * @return `K::cross_section(lhs, rhs, relative_speed)` in m^2.
     */
    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const K&) const noexcept { return K::cross_section(lhs, rhs, relative_speed); }
};

/**
 * @brief Visitor that applies any leaf's scatter operator to a captured velocity pair.
 *
 * Passed to @ref DsmcKernelVariant::apply; the template call operator forwards to the live
 * leaf instance and mutates the referenced velocities in place.
 */
class DsmcCollide {
public:
    Float3& lhs_velocity;          ///< First partner's velocity, updated by the leaf.
    Float3& rhs_velocity;          ///< Second partner's velocity, updated by the leaf.
    const Material& lhs;           ///< First partner's material.
    const Material& rhs;           ///< Second partner's material.
    default_random_engine& engine; ///< Variate source handed to the leaf; advanced in place.
    /**
     * @brief Runs the active leaf's scatter on the captured pair.
     * @tparam K The active kernel leaf type.
     * @param kernel The live leaf instance to invoke.
     */
    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const K& kernel) const noexcept { kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel() noexcept {
    // Activate the default hard-sphere arm through the variant so `type` and the union stay
    // consistent.
    DsmcKernelVariant::construct(*this, DsmcKernelType::hard_sphere);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const DsmcKernelType kernel_type) noexcept {
    DsmcKernelVariant::construct(*this, kernel_type);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::cross_section(const Material& lhs,
                          const Material& rhs,
                          const float relative_speed) const noexcept {
    // Fall back to 0 m^2 if the tag matches no arm, so an unset kernel simply collides nothing.
    return DsmcKernelVariant::visit(
        *this,
        DsmcCrossSection { lhs, rhs, relative_speed },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::sigma_g(const Material* materials,
                    const std::size_t lhs_species,
                    const std::size_t rhs_species,
                    const float relative_speed_squared) const noexcept {
    // A zero (or numerically negative) squared speed means no relative motion, hence no
    // collision flux; short-circuit before the square root.
    if (!(relative_speed_squared > 0.0f)) {
        return 0.0f;
    }

    const float relative_speed = atlas::sqrt_nonnegative(relative_speed_squared);

    return cross_section(materials[lhs_species], materials[rhs_species], relative_speed)
        * relative_speed;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcKernel::operator()(Float3& lhs_velocity,
                       Float3& rhs_velocity,
                       const Material& lhs,
                       const Material& rhs,
                       default_random_engine& engine) const noexcept {
    DsmcKernelVariant::apply(
        *this,
        DsmcCollide { lhs_velocity, rhs_velocity, lhs, rhs, engine });
}

}