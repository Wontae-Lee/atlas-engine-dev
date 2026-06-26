#include "../utilities/test_utils.h"

#include <atlas/scan/exclusive_scan.h>

#include <testkit/testkit.h>

#include <functional>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::exclusive_scan;

} // namespace

TEST(ExclusiveScan, SerialScanComputesPrefixSums) {
    // Arrange: create an input sequence and output buffer.
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    // Act: compute prefix sums with the serial policy.
    const auto end = exclusive_scan<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin(),
        0);

    // Assert: the output contains exclusive prefix sums.
    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 0, 1, 3, 6 }));
}

TEST(ExclusiveScan, HostScanComputesPrefixSums) {
    // Arrange: create an input sequence and output buffer.
    const std::vector<int> input { 2, 4, 6 };
    std::vector<int> output(input.size(), 0);

    // Act: compute prefix sums with the host policy.
    const auto end = exclusive_scan<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        output.begin(),
        1);

    // Assert: the output starts from the provided initial value.
    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 1, 3, 7 }));
}

TEST(ExclusiveScan, DeviceScanComputesPrefixSums) {
    // Arrange: create an input sequence and output buffer.
    const std::vector<int> input { 5, 1, 2 };
    std::vector<int> output(input.size(), 0);

    // Act: compute prefix sums with the device policy.
    const auto end = exclusive_scan<ExecutionPolicy::device>(
        input.begin(),
        input.end(),
        output.begin(),
        0);

    // Assert: the output contains exclusive prefix sums.
    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 0, 5, 6 }));
}

TEST(ExclusiveScan, CustomBinaryOperatorIsSupported) {
    // Arrange: create an input sequence and output buffer.
    const std::vector<int> input { 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    // Act: compute an exclusive scan with multiplication.
    exclusive_scan<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin(),
        1,
        std::multiplies<int> {});

    // Assert: the custom operator controls prefix accumulation.
    EXPECT_EQ(output, (std::vector<int> { 1, 2, 6 }));
}

TEST(ExclusiveScan, OverloadWithoutExplicitBinaryOperatorUsesPlus) {
    // Arrange: create an input sequence and output buffer.
    const std::vector<int> input { 3, 3, 3 };
    std::vector<int> output(input.size(), 0);

    // Act: compute a scan without passing an explicit operator.
    exclusive_scan<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin());

    // Assert: the overload uses addition with a zero initial value.
    EXPECT_EQ(output, (std::vector<int> { 0, 3, 6 }));
}

TEST(ExclusiveScan, EmptyRangeIsNoOp) {
    // Arrange: create an output buffer and scan an empty range.
    std::vector<int> output { 9, 9 };

    // Act: scan an empty range.
    const auto end = exclusive_scan<ExecutionPolicy::host>(
        output.begin(),
        output.begin(),
        output.begin(),
        0);

    // Assert: the iterator and output buffer are unchanged.
    EXPECT_EQ(end, output.begin());
    EXPECT_EQ(output, (std::vector<int> { 9, 9 }));
}
