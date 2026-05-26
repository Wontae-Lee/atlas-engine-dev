#pragma once

/**
 * @file dsmc_kernel.h
 * @brief Declares tagged DSMC collision kernels for runtime-selectable device-side dispatch.
 *
 * This header defines:
 * - `DsmcKernelType`, an enum describing which DSMC collision model is active
 * - `DsmcKernel<T>`, a tagged-union wrapper that stores exactly one concrete
 *   DSMC collision kernel and dispatches calls through a uniform interface
 *
 * The wrapper is designed for environments where:
 * - the active collision kernel must be selected at runtime
 * - the selected kernel must remain usable in host and device code
 * - dynamic polymorphism may be undesirable or unavailable
 *
 * Instead of virtual dispatch, this implementation uses an explicit type tag
 * plus a union of concrete kernel objects.
 */

#include <atlas/solver/dsmc/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <cstdint>

namespace atlas::system {

/**
 * @brief Identifies the concrete DSMC collision kernel stored in `DsmcKernel<T>`.
 *
 * Each enum value corresponds to one physically distinct collision model:
 * - `hard_sphere`:
 *   Constant-diameter hard-sphere model
 * - `variable_hard_sphere`:
 *   Velocity-dependent hard-sphere variant
 * - `variable_soft_sphere`:
 *   Velocity-dependent soft-sphere variant with additional scattering behavior
 */
enum struct DsmcKernelType : int {
    hard_sphere,
    variable_hard_sphere,
    variable_soft_sphere
};

/**
 * @brief Tagged runtime wrapper for DSMC collision kernels.
 *
 * `DsmcKernel<T>` stores exactly one active concrete collision kernel inside a
 * union and tracks the active member through the `type` tag.
 *
 * This design provides:
 * - compact storage
 * - explicit runtime dispatch without virtual functions
 * - compatibility with host/device execution paths
 *
 * Because a union stores non-trivial objects, this type must manage:
 * - construction of the active member
 * - destruction of the active member
 * - copying of the active member
 *
 * Therefore, custom constructors, assignment, and destructor logic are provided.
 *
 * @tparam T Floating-point scalar type used by the collision kernels.
 */
template <typename T>
struct DsmcKernel final {
    /**
     * @brief Type tag indicating which union member is currently active.
     */
    DsmcKernelType type = DsmcKernelType::hard_sphere;

    /**
     * @brief Storage for the active concrete kernel.
     *
     * Exactly one member is active at any given time, as indicated by `type`.
     */
    union {
        /**
         * @brief Hard-sphere collision kernel storage.
         */
        HardSphereKernel<T> hard_sphere;

        /**
         * @brief Variable hard-sphere collision kernel storage.
         */
        VariableHardSphereKernel<T> variable_hard_sphere;

        /**
         * @brief Variable soft-sphere collision kernel storage.
         */
        VariableSoftSphereKernel<T> variable_soft_sphere;
    };

    /**
     * @brief Construct a default kernel wrapper.
     *
     * The default active kernel is `DsmcKernelType::hard_sphere`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel() noexcept;

    /**
     * @brief Construct a kernel wrapper from a type tag.
     *
     * This constructor activates the concrete union member corresponding to
     * the provided kernel type.
     *
     * @param type Active kernel type to construct.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DsmcKernel(DsmcKernelType type) noexcept;

    /**
     * @brief Copy-construct a kernel wrapper.
     *
     * The new wrapper activates the same concrete kernel type as `other`
     * and copies the corresponding kernel object into its own union storage.
     *
     * @param other Source kernel wrapper.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel(const DsmcKernel& other) noexcept;

    /**
     * @brief Copy-assign a kernel wrapper.
     *
     * The current active kernel is destroyed, then the active kernel from
     * `other` is reconstructed and copied into this wrapper.
     *
     * @param other Source kernel wrapper.
     * @return Reference to this wrapper.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept;

    /**
     * @brief Destroy the active kernel stored in the union.
     *
     * Since the union may contain non-trivial kernel objects, destruction must
     * be dispatched manually according to the active `type`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DsmcKernel() noexcept;

    /**
     * @brief Construct a wrapper from a concrete hard-sphere kernel object.
     *
     * The wrapper tag is set to `DsmcKernelType::hard_sphere`, and the provided
     * kernel object becomes the active union member.
     *
     * @param op Concrete hard-sphere kernel object.
     */
    ATLAS_HOST
    DsmcKernel(const HardSphereKernel<T>& op);

    /**
     * @brief Construct a wrapper from a concrete variable hard-sphere kernel object.
     *
     * The wrapper tag is set to `DsmcKernelType::variable_hard_sphere`, and the
     * provided kernel object becomes the active union member.
     *
     * @param op Concrete variable hard-sphere kernel object.
     */
    ATLAS_HOST
    DsmcKernel(const VariableHardSphereKernel<T>& op);

    /**
     * @brief Construct a wrapper from a concrete variable soft-sphere kernel object.
     *
     * The wrapper tag is set to `DsmcKernelType::variable_soft_sphere`, and the
     * provided kernel object becomes the active union member.
     *
     * @param op Concrete variable soft-sphere kernel object.
     */
    ATLAS_HOST
    DsmcKernel(const VariableSoftSphereKernel<T>& op);

    /**
     * @brief Compute the collision cross section for a selected kernel type.
     *
     * This is a stateless convenience entry point that dispatches by explicit
     * `DsmcKernelType` rather than requiring a previously constructed wrapper.
     *
     * It is typically useful in contexts where only the kernel type and material
     * properties are needed to evaluate cross section behavior, without invoking
     * the full velocity-update collision operator.
     *
     * @param type Selected collision kernel type.
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Relative speed magnitude between the two particles.
     * @return Collision cross section for the selected kernel model.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(DsmcKernelType type,
                  const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Apply the active collision kernel to two particle velocities.
     *
     * This function dispatches to the currently active concrete kernel stored
     * in the union and updates the two particle velocities in place.
     *
     * The selected kernel may use:
     * - the current pre-collision velocities
     * - the material properties of both particles
     * - its own model-specific scattering / cross-section behavior
     *
     * @param lhs_velocity Velocity of the left-hand particle. Updated in place.
     * @param rhs_velocity Velocity of the right-hand particle. Updated in place.
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sigma_g(const MaterialProperties<T>* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            T relative_speed_squared) const noexcept;

public:
    /**
     * @brief Destroy the currently active union member.
     *
     * This helper must inspect `type` and invoke the destructor of the matching
     * concrete kernel stored in the union.
     *
     * It is used by:
     * - the wrapper destructor
     * - copy assignment before reconstructing a new active member
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Copy the active kernel state from another wrapper.
     *
     * This helper activates the same concrete kernel type as `other` and copies
     * its corresponding union member into this wrapper.
     *
     * It assumes that this function is responsible for correctly reconstructing
     * union storage according to `other.type`.
     *
     * @param other Source wrapper whose active kernel should be copied.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DsmcKernel& other) noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/dsmc_kernel.hpp>
