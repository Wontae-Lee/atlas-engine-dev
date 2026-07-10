#pragma once

namespace atlas {

/**
 * @brief Runtime tag selecting which collision-cross-section / scattering leaf a
 *        @ref DsmcKernel dispatches to.
 *
 * @ref DsmcKernel is a @ref DeviceVariant tagged union over the three leaf kernels
 * below; this enum is its discriminator. The order matters: it is the active-member
 * mapping used by @ref DsmcKernelVariant, and @c hard_sphere is the default arm the
 * union falls back to. Fixed to @c int so the tag is stable across serialization and
 * comparable on the device.
 */
enum class DsmcKernelType : int {

    hard_sphere, ///< Constant hard-sphere cross-section (@ref HardSphereKernel).

    variable_hard_sphere, ///< Temperature-dependent VHS cross-section (@ref VariableHardSphereKernel).

    variable_soft_sphere ///< VHS cross-section with soft-sphere angular scattering (@ref VariableSoftSphereKernel).
};

}