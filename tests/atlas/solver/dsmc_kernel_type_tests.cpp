#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::DsmcKernelType;

}

// Fixed to int so the tag is stable across serialization and comparable on the device.
static_assert(std::is_same_v<std::underlying_type_t<DsmcKernelType>, int>,
              "DsmcKernelType must be backed by int for host/device stability");

TEST(DsmcKernelType, EnumeratorsAreOrderedFromZero) {
    // The enumerator value doubles as the DeviceVariant active-member mapping used by
    // DsmcKernelVariant, and hard_sphere is the default fallback arm, so this order is a
    // stability contract, not an implementation detail.
    EXPECT_EQ(static_cast<int>(DsmcKernelType::hard_sphere), 0);
    EXPECT_EQ(static_cast<int>(DsmcKernelType::variable_hard_sphere), 1);
    EXPECT_EQ(static_cast<int>(DsmcKernelType::variable_soft_sphere), 2);
}

TEST(DsmcKernelType, RoundTripsThroughUnderlyingInt) {
    for (const DsmcKernelType type : { DsmcKernelType::hard_sphere,
                                       DsmcKernelType::variable_hard_sphere,
                                       DsmcKernelType::variable_soft_sphere }) {
        EXPECT_EQ(static_cast<DsmcKernelType>(static_cast<int>(type)), type);
    }
}
