#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <vector>

TEST(TransformReduce, EmptyRangeReturnsInit_Serial) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        7,
        [](const int x) { return x * 2; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(r, 7);
}

TEST(TransformReduce, EmptyRangeReturnsInit_Host) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        -3,
        [](const int x) { return x + 1; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(r, -3);
}

TEST(TransformReduce, EmptyRangeReturnsInit_Device) {
    const std::vector<int> v;

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        123,
        [](const int x) { return x; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(r, 123);
}

TEST(TransformReduce, SumSquares_MatchesExpected_Serial) {
    const std::vector<int> v { 1, 2, 3, 4 };

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        [](const int x) { return x * x; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(r, 30);
}

TEST(TransformReduce, SumSquares_HostMatchesSerial) {
    const std::vector<int> v { 1, 2, 3, 4, 5, 6, 7 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        [](const int x) { return x * x; },
        [](const int a, const int b) { return a + b; });

    const int host_r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        0,
        [](const int x) { return x * x; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(host_r, serial_r);
}

TEST(TransformReduce, SumSquares_DeviceMatchesSerial) {
    const std::vector<int> v { 3, 1, 4, 1, 5, 9 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        0,
        [](const int x) { return x * x; },
        [](const int a, const int b) { return a + b; });

    const int device_r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        0,
        [](const int x) { return x * x; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(device_r, serial_r);
}

TEST(TransformReduce, NonZeroInit_IsIncluded) {
    const std::vector<int> v { 1, 2, 3 };

    const int r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        10,
        [](const int x) { return x; },
        [](const int a, const int b) { return a + b; });

    EXPECT_EQ(r, 16);
}

TEST(TransformReduce, MultiplyReduce_WorksForAssociativeOp) {
    const std::vector<int> v { 1, 2, 3, 4 };

    const int serial_r = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(
        v.begin(),
        v.end(),
        1,
        [](const int x) { return x + 1; },
        [](const int a, const int b) { return a * b; });

    EXPECT_EQ(serial_r, 120);

    const int host_r = atlas::transform_reduce<atlas::ExecutionPolicy::host>(
        v.begin(),
        v.end(),
        1,
        [](const int x) { return x + 1; },
        [](const int a, const int b) { return a * b; });

    EXPECT_EQ(host_r, serial_r);

    const int device_r = atlas::transform_reduce<atlas::ExecutionPolicy::device>(
        v.begin(),
        v.end(),
        1,
        [](const int x) { return x + 1; },
        [](const int a, const int b) { return a * b; });

    EXPECT_EQ(device_r, serial_r);
}