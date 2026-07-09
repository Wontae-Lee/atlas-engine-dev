#pragma once

#include <atlas/core/macros.h>

namespace atlas {

struct Bool3 {
    bool x;

    bool y;

    bool z;
};

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const Bool3& b) noexcept {
    return b.x && b.y && b.z;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const Bool3& b) noexcept {
    return b.x || b.y || b.z;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
none(const Bool3& b) noexcept {
    return !(b.x || b.y || b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator&(const Bool3& a, const Bool3& b) noexcept {
    return { a.x && b.x, a.y && b.y, a.z && b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator|(const Bool3& a, const Bool3& b) noexcept {
    return { a.x || b.x, a.y || b.y, a.z || b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator!(const Bool3& b) noexcept {
    return { !b.x, !b.y, !b.z };
}

}