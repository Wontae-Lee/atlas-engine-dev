#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas::detail {

struct HitCollider final {

    float distance {};

    float speed {};

    Float3 point {};

    Float3 normal {};

    int unit_index { -1 };

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    found() const noexcept {
        return unit_index >= 0;
    }
};

}
