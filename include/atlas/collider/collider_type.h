#pragma once

namespace atlas {

/**
 * @brief Discriminator tag for the `Collider` tagged-union umbrella.
 *
 * One enumerator per collider leaf type. `Collider` stores a value of this enum
 * alongside its `union` of leaves; `ColliderVariant` (a `DeviceVariant`) reads
 * the tag to pick the active union member and dispatch `trace`/`collide`/
 * `advance`/`bound` to it on both host and device.
 *
 * The underlying type is fixed to `int` so the tag has a stable, trivially
 * copyable layout usable inside a `DeviceBuffer<Collider>` and on the device.
 *
 * @note To add a leaf, append a new enumerator here, add the matching union
 *       member and `DeviceVariantCase` to `ColliderVariant`, and a
 *       `static_assert(ConceptCollider<NewLeaf>)`.
 */
enum struct ColliderType : int {

    isothermal ///< `IsothermalCollider`: isothermal wall with specular/diffuse reflection.
};
}