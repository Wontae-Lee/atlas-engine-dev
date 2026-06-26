#include "../utilities/test_utils.h"

#include <atlas/transform/transform.h>

#include <testkit/testkit.h>

#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::transform;

} // namespace

TEST(Transform, UnaryTransformWorksForSerialHostAndDevice) {
    // Arrange: prepare a deterministic input and one output per policy.
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> serial_out(4, 0);
    std::vector<int> host_out(4, 0);
    std::vector<int> device_out(4, 0);

    // Act: double each element through every execution policy.
    transform<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        serial_out.begin(),
        [](int v) { return v * 2; });

    transform<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        host_out.begin(),
        [](int v) { return v * 2; });

    transform<ExecutionPolicy::device>(
        input.begin(),
        input.end(),
        device_out.begin(),
        [](int v) { return v * 2; });

    // Assert: all policies write the same transformed sequence.
    EXPECT_EQ(serial_out, (std::vector<int> { 2, 4, 6, 8 }));
    EXPECT_EQ(host_out, serial_out);
    EXPECT_EQ(device_out, serial_out);
}

TEST(Transform, BinaryTransformWorks) {
    // Arrange: prepare two equally sized input ranges.
    const std::vector<int> lhs { 1, 2, 3 };
    const std::vector<int> rhs { 4, 5, 6 };
    std::vector<int> out(3, 0);

    // Act: add matching elements with the host policy.
    transform<ExecutionPolicy::host>(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        out.begin(),
        [](int a, int b) { return a + b; });

    // Assert: each output element is the pairwise sum.
    EXPECT_EQ(out, (std::vector<int> { 5, 7, 9 }));
}

TEST(Transform, EmptyRangeIsNoOp) {
    // Arrange: prepare output storage that should remain untouched.
    std::vector<int> out { 1, 2 };

    // Act: transform an empty range.
    const auto end = transform<ExecutionPolicy::serial>(
        out.begin(),
        out.begin(),
        out.begin(),
        [](int v) { return v * 3; });

    // Assert: the returned iterator and output storage are unchanged.
    EXPECT_EQ(end, out.begin());
    EXPECT_EQ(out, (std::vector<int> { 1, 2 }));
}
