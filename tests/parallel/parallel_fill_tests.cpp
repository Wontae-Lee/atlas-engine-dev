#include "../utilities/tests_utils.h"

#include <atlas/parallel/parallel_fill.h>

#include <testkit/testkit.h>

#include <vector>

TEST(ParallelFill, SerialFillAssignsWholeRange) {
    std::vector<int> values(5, 0);

    atlas::parallel_fill<atlas::ExecutionPolicy::serial>(values.begin(), values.end(), 7);

    EXPECT_EQ(values, (std::vector<int> { 7, 7, 7, 7, 7 }));
}

TEST(ParallelFill, HostFillAssignsWholeRange) {
    std::vector<int> values(4, -1);

    atlas::parallel_fill<atlas::ExecutionPolicy::host>(values.begin(), values.end(), 3);

    EXPECT_EQ(values, (std::vector<int> { 3, 3, 3, 3 }));
}

TEST(ParallelFill, DeviceFillAssignsWholeRange) {
    std::vector<int> values(3, 1);

    atlas::parallel_fill<atlas::ExecutionPolicy::device>(values.begin(), values.end(), 9);

    EXPECT_EQ(values, (std::vector<int> { 9, 9, 9 }));
}

TEST(ParallelFill, EmptyRangeIsNoOp) {
    std::vector<int> values { 1, 2, 3 };

    EXPECT_NO_THROW(atlas::parallel_fill<atlas::ExecutionPolicy::host>(values.begin(), values.begin(), 5));
    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3 }));
}
