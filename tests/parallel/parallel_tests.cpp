#include <atlas/parallel/parallel.h>

#include <gtest/gtest.h>

TEST(Parallel, UmbrellaHeaderExposesExecutionPolicy) {
    constexpr auto host_policy   = atlas::ExecutionPolicy::host;
    constexpr auto device_policy = atlas::ExecutionPolicy::device;
    constexpr auto serial_policy = atlas::ExecutionPolicy::serial;

    EXPECT_NE(host_policy, device_policy);
    EXPECT_NE(host_policy, serial_policy);
    EXPECT_NE(device_policy, serial_policy);
}
