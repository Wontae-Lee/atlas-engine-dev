#include "../utilities/test_utils.h"

#include <atlas/transform/transform_reduce.h>

#include <testkit/testkit.h>

#include <functional>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::transform_reduce;

} // namespace

TEST(TransformReduce, ComputesMappedSumAcrossPolicies) {
    // Arrange: prepare a small deterministic input range.
    const std::vector<int> input { 1, 2, 3, 4 };

    // Act: double each value and reduce through every execution policy.
    const auto serial = transform_reduce<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        0,
        [](int v) { return v * 2; },
        std::plus<int> {});

    const auto host = transform_reduce<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        0,
        [](int v) { return v * 2; },
        std::plus<int> {});

    const auto device = transform_reduce<ExecutionPolicy::device>(
        input.begin(),
        input.end(),
        0,
        [](int v) { return v * 2; },
        std::plus<int> {});

    // Assert: all policies produce the same mapped sum.
    EXPECT_EQ(serial, 20);
    EXPECT_EQ(host, serial);
    EXPECT_EQ(device, serial);
}

TEST(TransformReduce, EmptyRangeReturnsInit) {
    // Arrange: prepare an empty input range.
    const std::vector<int> input;

    // Act: reduce the empty range with a non-zero initial value.
    const auto result = transform_reduce<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        7,
        [](int v) { return v; },
        std::plus<int> {});

    // Assert: empty reductions return the initial value.
    EXPECT_EQ(result, 7);
}
