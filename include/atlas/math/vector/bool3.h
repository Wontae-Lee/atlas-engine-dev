/**
 * @file   bool3.h
 * @brief  3-component boolean mask type Bool3 and related free functions.
 *
 * Bool3 holds the result of a component-wise comparison between two
 * Float3 vectors (see float3.h). It is a plain aggregate of three bools,
 * designed to work on both CPU (Host) and CUDA GPU (Device), and is
 * combined with logical AND/OR/NOT operators and reduced with
 * all() / any() / none().
 */

#pragma once

#include <atlas/core/macros.h> /**< ATLAS_ALL_DEVICE, ATLAS_FORCE_INLINE, ATLAS_NODISCARD */

namespace atlas {

/**
 * @struct Bool3
 * @brief 3-component boolean mask, typically produced by comparing two Float3 vectors.
 *
 * A plain aggregate (no user-declared constructors), so it supports
 * brace initialization such as `Bool3{true, false, true}`. Used as the
 * return type of Float3's component-wise comparison operators
 * (`operator<`, `operator<=`, `operator>`, `operator>=`) so a per-axis
 * result can be inspected or reduced instead of collapsing to a single
 * scalar bool.
 *
 * @code{.cpp}
 * atlas::Float3 a(1.f, 2.f, 3.f);
 * atlas::Float3 b(2.f, 2.f, 2.f);
 * atlas::Bool3 m = a < b;   // {true, false, false}
 * bool any_less  = atlas::any(m);
 * @endcode
 */
struct Bool3 {
    bool x; /**< First (X) component of the mask */

    bool y; /**< Second (Y) component of the mask */

    bool z; /**< Third (Z) component of the mask */
};

/**
 * @brief Checks whether all three components are true.
 * @param b  Mask to reduce
 * @return   b.x && b.y && b.z
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const Bool3& b) noexcept {
    return b.x && b.y && b.z;
}

/**
 * @brief Checks whether at least one component is true.
 * @param b  Mask to reduce
 * @return   b.x || b.y || b.z
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const Bool3& b) noexcept {
    return b.x || b.y || b.z;
}

/**
 * @brief Checks whether all three components are false.
 * @param b  Mask to reduce
 * @return   !(b.x || b.y || b.z)
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
none(const Bool3& b) noexcept {
    return !(b.x || b.y || b.z);
}

/**
 * @brief Component-wise logical AND of two masks.
 * @param a  Operand mask A
 * @param b  Operand mask B
 * @return   {a.x && b.x, a.y && b.y, a.z && b.z}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator&(const Bool3& a, const Bool3& b) noexcept {
    return { a.x && b.x, a.y && b.y, a.z && b.z };
}

/**
 * @brief Component-wise logical OR of two masks.
 * @param a  Operand mask A
 * @param b  Operand mask B
 * @return   {a.x || b.x, a.y || b.y, a.z || b.z}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator|(const Bool3& a, const Bool3& b) noexcept {
    return { a.x || b.x, a.y || b.y, a.z || b.z };
}

/**
 * @brief Component-wise logical NOT of a mask.
 * @param b  Mask to negate
 * @return   {!b.x, !b.y, !b.z}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator!(const Bool3& b) noexcept {
    return { !b.x, !b.y, !b.z };
}

} // namespace atlas
