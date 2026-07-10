#pragma once

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief A component-wise boolean triple, the result of comparing two 3-vectors.
 *
 * Produced by the relational operators on Float3 and Int3 (operator<, <=, >,
 * >=), which compare each axis independently. It is an aggregate with no
 * constructors so it can be brace-initialized on host or device and captured by
 * value in a device lambda. Collapse it to a single bool with all(), any(), or
 * none().
 */
struct Bool3 {
    bool x; ///< Result for the x component.

    bool y; ///< Result for the y component.

    bool z; ///< Result for the z component.
};

/**
 * @brief True when every component is true (logical AND across all axes).
 *
 * @param b The triple to reduce.
 * @return b.x && b.y && b.z.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const Bool3& b) noexcept {
    return b.x && b.y && b.z;
}

/**
 * @brief True when at least one component is true (logical OR across all axes).
 *
 * @param b The triple to reduce.
 * @return b.x || b.y || b.z.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const Bool3& b) noexcept {
    return b.x || b.y || b.z;
}

/**
 * @brief True when no component is true (the negation of any()).
 *
 * @param b The triple to reduce.
 * @return !(b.x || b.y || b.z).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
none(const Bool3& b) noexcept {
    return !(b.x || b.y || b.z);
}

/**
 * @brief Component-wise logical AND of two boolean triples.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return A Bool3 whose each axis is the AND of the corresponding inputs.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator&(const Bool3& a, const Bool3& b) noexcept {
    return { a.x && b.x, a.y && b.y, a.z && b.z };
}

/**
 * @brief Component-wise logical OR of two boolean triples.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return A Bool3 whose each axis is the OR of the corresponding inputs.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator|(const Bool3& a, const Bool3& b) noexcept {
    return { a.x || b.x, a.y || b.y, a.z || b.z };
}

/**
 * @brief Component-wise logical negation of a boolean triple.
 *
 * @param b The triple to negate.
 * @return A Bool3 with each axis flipped.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator!(const Bool3& b) noexcept {
    return { !b.x, !b.y, !b.z };
}

}