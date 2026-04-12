#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/solver/solver.h>

#include <type_traits>

namespace atlas::system {

enum class SphModelType : int {
    poly6,
    spiky,
    wendland
};

template <typename T>
struct SphPoly6Kernel final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SphPoly6Kernel(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct SphSpikyKernel final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SphSpikyKernel(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct SphWendlandKernel final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SphWendlandKernel(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct SphKernel final {
    static_assert(std::is_floating_point_v<T>, "SphKernel requires a floating-point T");

    SphModelType type = SphModelType::spiky;
    union {
        SphPoly6Kernel<T> poly6;
        SphSpikyKernel<T> spiky;
        SphWendlandKernel<T> wendland;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel(SphModelType type,
                T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel(const SphKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SphKernel&
    operator=(const SphKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SphKernel() noexcept;

    ATLAS_HOST
    SphKernel(const SphPoly6Kernel<T>& op);

    ATLAS_HOST
    SphKernel(const SphSpikyKernel<T>& op);

    ATLAS_HOST
    SphKernel(const SphWendlandKernel<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SphKernel& other) noexcept;
};

}

namespace atlas {

template <typename T>
using SphOperator = atlas::system::SphKernel<T>;

template <typename T>
using Poly6SphOperator = atlas::system::SphPoly6Kernel<T>;

template <typename T>
using SpikySphOperator = atlas::system::SphSpikyKernel<T>;

template <typename T>
using WendlandSphOperator = atlas::system::SphWendlandKernel<T>;

using SphModelType = atlas::system::SphModelType;

}

#include <atlas/solver/sph/sph_kernel.hpp>
