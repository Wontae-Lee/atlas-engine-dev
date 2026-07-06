#include <atlas/scan/exclusive_scan.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <functional>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::exclusive_scan;

}

TEST(ExclusiveScan, SerialScanComputesPrefixSums) {
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    const auto end = exclusive_scan<ExecutionPolicy::serial>(
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

    const auto end = exclusive_scan<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        output.begin(),
        1);

    EXPECT_EQ(end, output.end());
    EXPECT_EQ(output, (std::vector<int> { 1, 3, 7 }));
}

TEST(ExclusiveScan, DeviceScanComputesPrefixSums) {
    const std::vector<int> input { 5, 1, 2 };
    atlas::DeviceBuffer<int> device_input(input.size());
    atlas::DeviceBuffer<int> device_output(input.size(), 0);
    atlas::copy_host_to_device(input.data(), device_input, input.size());

    exclusive_scan<ExecutionPolicy::device>(
        device_input.begin(),
        device_input.end(),
        device_output.begin(),
        0);

    std::vector<int> output(input.size(), 0);
    atlas::copy_device_to_host(device_output, output.data(), output.size());

    EXPECT_EQ(output, (std::vector<int> { 0, 5, 6 }));
}

TEST(ExclusiveScan, CustomBinaryOperatorIsSupported) {
    const std::vector<int> input { 2, 3, 4 };
    std::vector<int> output(input.size(), 0);

    exclusive_scan<ExecutionPolicy::serial>(
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

    exclusive_scan<ExecutionPolicy::serial>(
        input.begin(),
        input.end(),
        output.begin());

    EXPECT_EQ(output, (std::vector<int> { 0, 3, 6 }));
}

TEST(ExclusiveScan, EmptyRangeIsNoOp) {
    std::vector<int> output { 9, 9 };

    const auto end = exclusive_scan<ExecutionPolicy::host>(
        output.begin(),
        output.begin(),
        output.begin(),
        0);

    EXPECT_EQ(end, output.begin());
    EXPECT_EQ(output, (std::vector<int> { 9, 9 }));
}
