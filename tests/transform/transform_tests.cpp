#include "../utilities/tests_utils.h"

#include <atlas/transform/transform.h>

#include <testkit/testkit.h>

#include <vector>

TEST(Transform, UnaryTransformWorksForSerialHostAndDevice) {
    const std::vector<int> input { 1, 2, 3, 4 };
    std::vector<int> serial_out(4, 0);
    std::vector<int> host_out(4, 0);
    std::vector<int> device_out(4, 0);

    atlas::transform<atlas::ExecutionPolicy::serial>(input.begin(), input.end(), serial_out.begin(), [](int v) { return v * 2; });
    atlas::transform<atlas::ExecutionPolicy::host>(input.begin(), input.end(), host_out.begin(), [](int v) { return v * 2; });
    atlas::transform<atlas::ExecutionPolicy::device>(input.begin(), input.end(), device_out.begin(), [](int v) { return v * 2; });

    EXPECT_EQ(serial_out, (std::vector<int> { 2, 4, 6, 8 }));
    EXPECT_EQ(host_out, serial_out);
    EXPECT_EQ(device_out, serial_out);
}

TEST(Transform, BinaryTransformWorks) {
    const std::vector<int> lhs { 1, 2, 3 };
    const std::vector<int> rhs { 4, 5, 6 };
    std::vector<int> out(3, 0);

    atlas::transform<atlas::ExecutionPolicy::host>(lhs.begin(), lhs.end(), rhs.begin(), out.begin(), [](int a, int b) { return a + b; });

    EXPECT_EQ(out, (std::vector<int> { 5, 7, 9 }));
}

TEST(Transform, EmptyRangeIsNoOp) {
    std::vector<int> out { 1, 2 };

    const auto end = atlas::transform<atlas::ExecutionPolicy::serial>(out.begin(), out.begin(), out.begin(), [](int v) { return v * 3; });

    EXPECT_EQ(end, out.begin());
    EXPECT_EQ(out, (std::vector<int> { 1, 2 }));
}
