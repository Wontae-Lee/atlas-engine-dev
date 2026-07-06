#include <atlas/transform/transform.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

using atlas::ExecutionPolicy;
using atlas::transform;

struct DoubleOp {
    ATLAS_ALL_DEVICE int
    operator()(const int v) const {
        return v * 2;
    }
};

}

TEST(Transform, UnaryTransformWorksForSerialAndHost) {
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> serial_out(4, 0);
    std::vector<int> host_out(4, 0);

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

    EXPECT_EQ(serial_out, (std::vector<int> { 2, 4, 6, 8 }));
    EXPECT_EQ(host_out, serial_out);
}

TEST(Transform, UnaryTransformWorksForDevice) {
    const std::vector<int> input { 1, 2, 3, 4 };
    atlas::DeviceBuffer<int> device_input(input.size());
    atlas::DeviceBuffer<int> device_output(input.size(), 0);
    atlas::copy_host_to_device(input.data(), device_input, input.size());

    transform<ExecutionPolicy::device>(
        device_input.begin(),
        device_input.end(),
        device_output.begin(),
        DoubleOp {});

    std::vector<int> output(input.size(), 0);
    atlas::copy_device_to_host(device_output, output.data(), output.size());

    EXPECT_EQ(output, (std::vector<int> { 2, 4, 6, 8 }));
}

TEST(Transform, BinaryTransformWorks) {
    const std::vector<int> lhs { 1, 2, 3 };
    const std::vector<int> rhs { 4, 5, 6 };
    std::vector<int> out(3, 0);

    transform<ExecutionPolicy::host>(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        out.begin(),
        [](int a, int b) { return a + b; });

    EXPECT_EQ(out, (std::vector<int> { 5, 7, 9 }));
}

TEST(Transform, EmptyRangeIsNoOp) {
    std::vector<int> out { 1, 2 };

    const auto end = transform<ExecutionPolicy::serial>(
        out.begin(),
        out.begin(),
        out.begin(),
        [](int v) { return v * 3; });

    EXPECT_EQ(end, out.begin());
    EXPECT_EQ(out, (std::vector<int> { 1, 2 }));
}
