#pragma once

/**
 * @file collider_hit.h
 * @brief Defines the selected surface hit used by collider collision kernels.
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas::detail {

/**
 * @brief Closest collider hit selected for one particle sweep.
 *
 * @tparam T Floating-point scalar type used by the collider.
 */
template <typename T>
struct ColliderHit final {
    T distance {};
    T speed {};
    Vector3<T> position {};
    Vector3<T> normal {};
    int unit_index { -1 };

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    found() const noexcept {
        return unit_index >= 0;
    }
};

} // namespace atlas::detail
