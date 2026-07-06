#include <atlas/transform/transform_reduce.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <functional>
#include <thrust/functional.h>
#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::transform_reduce;

struct DoubleOp {
    ATLAS_ALL_DEVICE int
    operator()(const int v) const {
        return v * 2;
    }
};

}

TEST(TransformReduce, ComputesMappedSumForSerialAndHost) {
    const std::vector<int> input { 1, 2, 3, 4 };

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

    EXPECT_EQ(serial, 20);
    EXPECT_EQ(host, serial);
}

TEST(TransformReduce, ComputesMappedSumForDevice) {
    const std::vector<int> input { 1, 2, 3, 4 };
    atlas::DeviceBuffer<int> device_input(input.size());
    atlas::copy_host_to_device(input.data(), device_input, input.size());

    const auto device = transform_reduce<ExecutionPolicy::device>(
        device_input.begin(),
        device_input.end(),
        0,
        DoubleOp {},
        thrust::plus<int> {});

    EXPECT_EQ(device, 20);
}

TEST(TransformReduce, EmptyRangeReturnsInit) {
    const std::vector<int> input;

    const auto result = transform_reduce<ExecutionPolicy::host>(
        input.begin(),
        input.end(),
        7,
        [](int v) { return v; },
        std::plus<int> {});

    EXPECT_EQ(result, 7);
}
