#include "../utilities/tests_utils.h"

#include <atlas/transform/transform_reduce.h>

#include <testkit/testkit.h>

#include <functional>
#include <vector>

TEST(TransformReduce, ComputesMappedSumAcrossPolicies) {
    const std::vector<int> input { 1, 2, 3, 4 };

    const auto serial = atlas::transform_reduce<atlas::ExecutionPolicy::serial>(input.begin(), input.end(), 0, [](int v) { return v * 2; }, std::plus<int> {});
    const auto host = atlas::transform_reduce<atlas::ExecutionPolicy::host>(input.begin(), input.end(), 0, [](int v) { return v * 2; }, std::plus<int> {});
    const auto device = atlas::transform_reduce<atlas::ExecutionPolicy::device>(input.begin(), input.end(), 0, [](int v) { return v * 2; }, std::plus<int> {});

    EXPECT_EQ(serial, 20);
    EXPECT_EQ(host, serial);
    EXPECT_EQ(device, serial);
}

TEST(TransformReduce, EmptyRangeReturnsInit) {
    const std::vector<int> input;

    const auto result = atlas::transform_reduce<atlas::ExecutionPolicy::host>(input.begin(), input.end(), 7, [](int v) { return v; }, std::plus<int> {});

    EXPECT_EQ(result, 7);
}
