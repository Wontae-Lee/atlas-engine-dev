#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/sph/cubic_spline_sph_kernel.h>
#include <atlas/solver/sph/standard_sph_kernel.h>
#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

#include <type_traits>

/**
 * @file sph_kernel.h
 * @brief Runtime-selectable dispatcher over the three SPH smoothing
 *        kernels (`StandardSphKernel`, `CubicSplineSphKernel`,
 *        `WendlandQuinticSphKernel`), analogous to `DsmcKernel` for the
 *        DSMC collision models.
 *
 * @details
 * ### Background — SPH kernel interpolation
 * Smoothed Particle Hydrodynamics represents a continuum field (density,
 * pressure, velocity) as a weighted sum over nearby particles, with the
 * weight given by a compactly-supported *smoothing kernel* `W(r, h)`
 * (`r` = distance to a neighbor, `h` = the kernel's support radius —
 * `cell_size` in this file's kernels). The kernel interpolant and its
 * derivatives give the three SPH field operators every solver needs:
 * ```
 * rho_i         = sum_j  m_j * W(r_ij, h)                     (density)
 * grad(P)_i     ~ sum_j  m_j * (...) * grad(W)(r_ij, h)        (pressure force)
 * laplacian(v)_i ~ sum_j m_j * (...) * laplacian(W)(r_ij, h)   (viscosity force)
 * ```
 * (see `SphSolver` for the full force assembly `(...)` terms — this file
 * only concerns the kernel shape itself). Every valid SPH kernel must be
 * normalized (`integral W dV = 1`), compactly supported (`W = 0` for
 * `r > h`, so only nearby particles contribute — what makes the sum
 * tractable), and smooth enough that its needed derivatives exist.
 * Different kernel families trade smoothness, computational cost, and
 * resistance to particle clustering ("pairing instability") differently
 * — see each kernel's own file for its specific shape and origin.
 * `StandardSphKernel` follows Müller et al.'s original real-time-graphics
 * SPH construction of using a *different*, specially-shaped kernel per
 * operator (density/pressure/viscosity) to avoid known failure modes of
 * differentiating one kernel three ways; `CubicSplineSphKernel` and
 * `WendlandQuinticSphKernel` instead use the literal analytic gradient
 * and Laplacian of a single kernel function, the more standard approach
 * in physically-rigorous (e.g. astrophysical) SPH.
 *
 * ### Operating principle
 * Same tagged-union `DeviceVariant` dispatch pattern as `DsmcKernel` (see
 * that file): an `SphKernelType` tag selects which kernel shape's
 * `density_weight`/`pressure_gradient`/`viscosity_laplacian` static
 * methods `SphKernel` forwards to, both via the static
 * `SphKernel::density_weight(type, ...)` overloads (no instance needed)
 * and the instance methods that read `this->type`.
 */

namespace atlas {

/**
 * @brief Selects which SPH smoothing kernel an `SphKernel`/`SphSolver`
 *        uses; see `standard_sph_kernel.h`, `cubic_spline_sph_kernel.h`,
 *        `wendland_quintic_sph_kernel.h` for each kernel's formula and
 *        origin.
 */
enum struct SphKernelType : int {
    /** Müller et al. (2003) Poly6/Spiky/viscosity kernel trio. */
    standard,
    /** Monaghan (1992) cubic B-spline (M4), single consistent kernel. */
    cubic_spline,
    /** Wendland (1995) C2 quintic, single consistent kernel, strong
     *  pairing-instability resistance. */
    wendland_quintic
};

/**
 * @brief Tagged-union wrapper letting `SphSolver` evaluate whichever
 *        `SphKernelType` it was configured with, without virtual
 *        dispatch. See this file's top-of-file documentation for the
 *        SPH kernel-interpolation background and the `DeviceVariant`
 *        pattern.
 */
struct SphKernel final {

    SphKernelType type = SphKernelType::standard;

    union {

        StandardSphKernel standard;

        CubicSplineSphKernel cubic_spline;

        WendlandQuinticSphKernel wendland_quintic;
    };

    ATLAS_ALL_DEVICE
    SphKernel() noexcept;

    ATLAS_ALL_DEVICE explicit SphKernel(SphKernelType type) noexcept;

    ATLAS_ALL_DEVICE
    SphKernel(const SphKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE SphKernel&
    operator=(const SphKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE ~SphKernel() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit SphKernel(const Payload& op);

    /** @brief `W(radius, cell_size)` for the given `type` — the density
     *  interpolation weight; see this file's top-of-file documentation. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    density_weight(SphKernelType type,
                   float radius,
                   float cell_size) noexcept;

    /** @brief `grad(W)(radius, cell_size)` along `delta` for the given
     *  `type` — the pressure-force gradient term. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3
    pressure_gradient(SphKernelType type,
                      const Vector3& delta,
                      float radius,
                      float cell_size) noexcept;

    /** @brief `laplacian(W)(radius, cell_size)` for the given `type` —
     *  the viscosity-force term. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    viscosity_laplacian(SphKernelType type,
                        float radius,
                        float cell_size) noexcept;

    /** @brief Instance form of `density_weight`, using `this->type`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    density_weight(float radius, float cell_size) const noexcept;

    /** @brief Instance form of `pressure_gradient`, using `this->type`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    pressure_gradient(const Vector3& delta,
                      float radius,
                      float cell_size) const noexcept;

    /** @brief Instance form of `viscosity_laplacian`, using `this->type`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    viscosity_laplacian(float radius, float cell_size) const noexcept;
};

namespace detail {

    using SphKernelVariant = DeviceVariant<
        SphKernel,
        SphKernelType,
        SphKernelType::standard,
        DeviceVariantCase<SphKernelType::standard, &SphKernel::standard>,
        DeviceVariantCase<SphKernelType::cubic_spline, &SphKernel::cubic_spline>,
        DeviceVariantCase<SphKernelType::wendland_quintic, &SphKernel::wendland_quintic>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
    struct SphDensityWeight {
        float radius;
        float cell_size;
        template <typename Tag> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
        operator()(Tag) const noexcept {
            using Kernel = typename Tag::type;
            return Kernel::density_weight(radius, cell_size);
        }
    };
    struct SphPressureGradient {
        const Vector3& delta;
        float radius;
        float cell_size;
        template <typename Tag> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
        operator()(Tag) const noexcept {
            using Kernel = typename Tag::type;
            return Kernel::pressure_gradient(delta, radius, cell_size);
        }
    };
    struct SphViscosityLaplacian {
        float radius;
        float cell_size;
        template <typename Tag> ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
        operator()(Tag) const noexcept {
            using Kernel = typename Tag::type;
            return Kernel::viscosity_laplacian(radius, cell_size);
        }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel() noexcept {
    detail::SphKernelVariant::construct(*this, SphKernelType::standard);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel(const SphKernelType type) noexcept {
    detail::SphKernelVariant::construct(*this, type);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel(const Payload& op) {
    detail::SphKernelVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::density_weight(const SphKernelType type,
                          const float radius,
                          const float cell_size) noexcept {
    return detail::SphKernelVariant::visit_type(
        type, detail::SphDensityWeight { radius, cell_size }, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
SphKernel::pressure_gradient(const SphKernelType type,
                             const Vector3& delta,
                             const float radius,
                             const float cell_size) noexcept {
    return detail::SphKernelVariant::visit_type(
        type, detail::SphPressureGradient { delta, radius, cell_size }, Vector3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::viscosity_laplacian(const SphKernelType type,
                               const float radius,
                               const float cell_size) noexcept {
    return detail::SphKernelVariant::visit_type(
        type, detail::SphViscosityLaplacian { radius, cell_size }, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::density_weight(const float radius, const float cell_size) const noexcept {
    return density_weight(type, radius, cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
SphKernel::pressure_gradient(const Vector3& delta,
                             const float radius,
                             const float cell_size) const noexcept {
    return pressure_gradient(type, delta, radius, cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::viscosity_laplacian(const float radius, const float cell_size) const noexcept {
    return viscosity_laplacian(type, radius, cell_size);
}

}
