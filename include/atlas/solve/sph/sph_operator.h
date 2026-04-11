#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/solve/solve.h>

#include <type_traits>

namespace atlas::system {

enum class SphModelType : int {
    poly6,
    spiky,
    wendland
};

template <typename T>
struct Poly6SphOperator final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Poly6SphOperator(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct SpikySphOperator final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SpikySphOperator(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct WendlandSphOperator final {
    T support_scale { T(1) };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit WendlandSphOperator(T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    weight(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    gradient_factor(T distance, T smoothing_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    laplacian(T distance, T smoothing_length) const noexcept;
};

template <typename T>
struct SphOperator final {
    static_assert(std::is_floating_point_v<T>, "SphOperator requires a floating-point T");

    SphModelType type = SphModelType::spiky;
    union {
        Poly6SphOperator<T> poly6;
        SpikySphOperator<T> spiky;
        WendlandSphOperator<T> wendland;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphOperator(SphModelType type,
                T support_scale = T(1)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphOperator(const SphOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SphOperator&
    operator=(const SphOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SphOperator() noexcept;

    ATLAS_HOST
    SphOperator(const Poly6SphOperator<T>& op);

    ATLAS_HOST
    SphOperator(const SpikySphOperator<T>& op);

    ATLAS_HOST
    SphOperator(const WendlandSphOperator<T>& op);

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
    copy_from(const SphOperator& other) noexcept;
};

}

namespace atlas {

template <typename T>
using SphOperator = atlas::system::SphOperator<T>;

template <typename T>
using Poly6SphOperator = atlas::system::Poly6SphOperator<T>;

template <typename T>
using SpikySphOperator = atlas::system::SpikySphOperator<T>;

template <typename T>
using WendlandSphOperator = atlas::system::WendlandSphOperator<T>;

using SphModelType = atlas::system::SphModelType;

}

#include <atlas/solve/sph/sph_operator.hpp>
