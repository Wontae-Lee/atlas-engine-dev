#include "../utilities/tests_utils.h"

#include <atlas/scan/exclusive_scan.h>

#include <testkit/testkit.h>

#include <functional>
#include <vector>

TEST(ExclusiveScan, SerialScanComputesPrefixSums) {
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    const auto end = atlas::exclusive_scan<atlas::ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin(),
        0);

    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 0, 1, 3, 6 }));
}

TEST(ExclusiveScan, HostScanComputesPrefixSums) {
    const std::vector<int> input { 2, 4, 6 };
    std::vector<int> output(input.size(), 0);

    const auto end = atlas::exclusive_scan<atlas::ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        output.begin(),
        1);

    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 1, 3, 7 }));
}

TEST(ExclusiveScan, DeviceScanComputesPrefixSums) {
    const std::vector<int> input { 5, 1, 2 };
    std::vector<int> output(input.size(), 0);

    const auto end = atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        input.begin(),
        input.end(),
        output.begin(),
        0);

    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 0, 5, 6 }));
}

TEST(ExclusiveScan, CustomBinaryOperatorIsSupported) {
    const std::vector<int> input { 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    atlas::exclusive_scan<atlas::ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin(),
        1,
        std::multiplies<int> {});

    EXPECT_EQ(output, (std::vector<int> { 1, 2, 6 }));
}

TEST(ExclusiveScan, OverloadWithoutExplicitBinaryOperatorUsesPlus) {
    const std::vector<int> input { 3, 3, 3 };
    std::vector<int> output(input.size(), 0);

    atlas::exclusive_scan<atlas::ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin());

    EXPECT_EQ(output, (std::vector<int> { 0, 3, 6 }));
}

TEST(ExclusiveScan, EmptyRangeIsNoOp) {
    std::vector<int> output { 9, 9 };

    const auto end = atlas::exclusive_scan<atlas::ExecutionPolicy::host>(
        output.begin(),
        output.begin(),
        output.begin(),
        0);

    EXPECT_EQ(end, output.begin());
    EXPECT_EQ(output, (std::vector<int> { 9, 9 }));
}
