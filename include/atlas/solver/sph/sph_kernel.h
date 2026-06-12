#pragma once

/**
 * @file sph_kernel.h
 * @brief Declares tagged SPH smoothing kernels for runtime dispatch.
 *
 * This header defines:
 * - `SphKernelType`, which identifies the active SPH smoothing-kernel model
 * - `SphKernel<T>`, a tagged-union wrapper that stores exactly one concrete
 *   SPH kernel and dispatches calls through a uniform runtime interface
 *
 * The wrapper is intended for cases where:
 * - the SPH kernel must be selected at runtime
 * - the selected kernel must remain usable in host and device code
 * - dynamic polymorphism is undesirable or unavailable
 *
 * Instead of virtual dispatch, this implementation uses:
 * - an explicit kernel-type tag
 * - a union holding one concrete kernel object
 *
 * The supported kernel families are:
 * - `StandardSphKernel<T>`
 * - `CubicSplineSphKernel<T>`
 * - `WendlandQuinticSphKernel<T>`
 */

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/sph/cubic_spline_sph_kernel.h>
#include <atlas/solver/sph/standard_sph_kernel.h>
#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

namespace atlas {

/**
 * @brief Identifies the concrete SPH smoothing kernel stored in `SphKernel<T>`.
 *
 * Each enumerator corresponds to one supported kernel family:
 * - `standard`:
 *   Baseline SPH kernel implementation used by the runtime
 * - `cubic_spline`:
 *   Compact-support cubic spline kernel commonly used in SPH
 * - `wendland_quintic`:
 *   Wendland quintic kernel with improved smoothness and stability properties
 */
enum struct SphKernelType : int {
    standard,
    cubic_spline,
    wendland_quintic
};

/**
 * @brief Tagged runtime wrapper for SPH smoothing kernels.
 *
 * `SphKernel<T>` stores exactly one active concrete SPH kernel inside a union
 * and tracks the active member with the `type` tag.
 *
 * This design provides:
 * - compact storage
 * - explicit runtime dispatch without virtual methods
 * - host/device compatibility
 *
 * Since the wrapper uses a union of potentially non-trivial types, it must manage:
 * - explicit construction of the active member
 * - explicit destruction of the active member
 * - explicit copying of the active member
 *
 * For that reason, this type provides custom constructors, assignment, and
 * destruction helpers.
 *
 * The wrapper exposes both:
 * - static dispatch helpers that take an explicit `SphKernelType`
 * - instance methods that dispatch using the currently active kernel
 *
 * @tparam T Floating-point scalar type used by the kernel implementation.
 */
template <typename T>
struct SphKernel final {
    /**
     * @brief Tag describing which union member is currently active.
     */
    SphKernelType type = SphKernelType::standard;

    /**
     * @brief Storage for the active concrete SPH kernel.
     *
     * Exactly one member is active at any given time, as specified by `type`.
     */
    union {
        /**
         * @brief Storage for the standard SPH kernel.
         */
        StandardSphKernel<T> standard;

        /**
         * @brief Storage for the cubic spline SPH kernel.
         */
        CubicSplineSphKernel<T> cubic_spline;

        /**
         * @brief Storage for the Wendland quintic SPH kernel.
         */
        WendlandQuinticSphKernel<T> wendland_quintic;
    };

    /**
     * @brief Construct a default SPH kernel wrapper.
     *
     * The default active kernel type is `SphKernelType::standard`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel() noexcept;

    /**
     * @brief Construct a kernel wrapper from a kernel-type tag.
     *
     * This constructor activates the concrete union member corresponding to the
     * given `type`.
     *
     * @param type Kernel type to activate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SphKernel(SphKernelType type) noexcept;

    /**
     * @brief Copy-construct a kernel wrapper.
     *
     * The newly constructed wrapper activates the same concrete kernel type as
     * `other` and copies the corresponding kernel object into its own union storage.
     *
     * @param other Source wrapper to copy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel(const SphKernel& other) noexcept;

    /**
     * @brief Copy-assign a kernel wrapper.
     *
     * The current active union member is destroyed, then the active kernel stored
     * in `other` is reconstructed and copied into this wrapper.
     *
     * @param other Source wrapper to copy from.
     * @return Reference to this wrapper.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SphKernel&
    operator=(const SphKernel& other) noexcept;

    /**
     * @brief Destroy the currently active kernel stored in the union.
     *
     * Because the wrapper uses tagged union storage, destruction must be routed
     * manually to the active member indicated by `type`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SphKernel() noexcept;

    /**
     * @brief Construct a wrapper from a concrete standard SPH kernel object.
     *
     * The wrapper tag is set to `SphKernelType::standard`, and the provided
     * kernel object becomes the active union member.
     *
     * @param op Concrete standard SPH kernel.
     */
    ATLAS_HOST
    SphKernel(const StandardSphKernel<T>& op);

    /**
     * @brief Construct a wrapper from a concrete cubic spline SPH kernel object.
     *
     * The wrapper tag is set to `SphKernelType::cubic_spline`, and the provided
     * kernel object becomes the active union member.
     *
     * @param op Concrete cubic spline SPH kernel.
     */
    ATLAS_HOST
    SphKernel(const CubicSplineSphKernel<T>& op);

    /**
     * @brief Construct a wrapper from a concrete Wendland quintic SPH kernel object.
     *
     * The wrapper tag is set to `SphKernelType::wendland_quintic`, and the provided
     * kernel object becomes the active union member.
     *
     * @param op Concrete Wendland quintic SPH kernel.
     */
    ATLAS_HOST
    SphKernel(const WendlandQuinticSphKernel<T>& op);

    /**
     * @brief Evaluate the scalar density weight for a selected kernel type.
     *
     * This is a stateless dispatch helper that selects the kernel implementation
     * using an explicit `SphKernelType` rather than relying on a previously
     * constructed wrapper instance.
     *
     * This function is typically used in density accumulation terms such as:
     * \f[
     * \rho_i = \sum_j m_j W(r_{ij}, h)
     * \f]
     *
     * @param type Selected SPH kernel type.
     * @param radius Inter-particle distance.
     * @param cell_size SPH cell size.
     * @return Scalar density kernel weight.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(SphKernelType type,
                   T radius,
                   T cell_size) noexcept;

    /**
     * @brief Evaluate the pressure-gradient term for a selected kernel type.
     *
     * This is a stateless dispatch helper that returns the gradient of the chosen
     * smoothing kernel using an explicit `SphKernelType`.
     *
     * It is typically used in pressure-force terms that depend on:
     * - the relative displacement vector `delta`
     * - the scalar distance `radius`
     * - the cell size
     *
     * @param type Selected SPH kernel type.
     * @param delta Relative displacement vector between two particles.
     * @param radius Magnitude of `delta`.
     * @param cell_size SPH cell size.
     * @return Pressure-gradient vector for the selected kernel.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(SphKernelType type,
                      const Vector3<T>& delta,
                      T radius,
                      T cell_size) noexcept;

    /**
     * @brief Evaluate the viscosity Laplacian for a selected kernel type.
     *
     * This is a stateless dispatch helper that selects the kernel implementation
     * by explicit type and returns the scalar Laplacian term typically used in
     * viscosity-force formulations.
     *
     * @param type Selected SPH kernel type.
     * @param radius Inter-particle distance.
     * @param cell_size SPH cell size.
     * @return Scalar viscosity Laplacian value.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(SphKernelType type,
                        T radius,
                        T cell_size) noexcept;

    /**
     * @brief Evaluate the scalar density weight using the currently active kernel.
     *
     * This instance method dispatches through the active kernel stored in the union.
     *
     * @param radius Inter-particle distance.
     * @param cell_size SPH cell size.
     * @return Scalar density kernel weight.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    density_weight(T radius, T cell_size) const noexcept;

    /**
     * @brief Evaluate the pressure-gradient term using the currently active kernel.
     *
     * This instance method dispatches through the active kernel stored in the union.
     *
     * @param delta Relative displacement vector between two particles.
     * @param radius Magnitude of `delta`.
     * @param cell_size SPH cell size.
     * @return Pressure-gradient vector.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T cell_size) const noexcept;

    /**
     * @brief Evaluate the viscosity Laplacian using the currently active kernel.
     *
     * This instance method dispatches through the active kernel stored in the union.
     *
     * @param radius Inter-particle distance.
     * @param cell_size SPH cell size.
     * @return Scalar viscosity Laplacian value.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    viscosity_laplacian(T radius, T cell_size) const noexcept;

private:
    /**
     * @brief Destroy the currently active kernel object stored in the union.
     *
     * This helper is responsible for invoking the correct destructor according to
     * the active `type` tag.
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
     * This helper reconstructs the correct union member according to `other.type`
     * and copies the corresponding kernel object into this wrapper.
     *
     * @param other Source wrapper whose active kernel should be copied.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SphKernel& other) noexcept;
};

} // namespace atlas

#include <atlas/solver/sph/sph_kernel.hpp>
