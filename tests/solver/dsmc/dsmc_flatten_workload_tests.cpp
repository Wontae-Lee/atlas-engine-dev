#include "dsmc_test_utils.h"

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>

#include <testkit/testkit.h>

TEST(DsmcFlattenWorkload, ClearResetsAllBuffersAndCount) {
    atlas::DsmcFlattenWorkload<float> workload;
    atlas::DeviceBuffer<int> counts(2);
    counts[0] = 1;
    counts[1] = 2;

    ASSERT_TRUE(workload.build(atlas::raw_pointer_cast(counts.data()), 2));
    workload.clear();

    EXPECT_TRUE(workload.offsets().empty());
    EXPECT_TRUE(workload.collision_cells.empty());
    EXPECT_TRUE(workload.filtered_collision_counts.empty());
    EXPECT_EQ(workload.flattened_collision_count, 0);
}

TEST(DsmcFlattenWorkload, BuildRejectsEmptyInput) {
    atlas::DsmcFlattenWorkload<float> workload;

    EXPECT_FALSE(workload.build(nullptr, 0));
    EXPECT_TRUE(workload.offsets().empty());
    EXPECT_EQ(workload.flattened_collision_count, 0);
}

TEST(DsmcFlattenWorkload, BuildFlattensPerCellCollisionCounts) {
    atlas::DsmcFlattenWorkload<float> workload;
    atlas::DeviceBuffer<int> counts(3);
    counts[0] = 2;
    counts[1] = 0;
    counts[2] = 1;

    ASSERT_TRUE(workload.build(atlas::raw_pointer_cast(counts.data()), 3));

    EXPECT_EQ(workload.offsets().size(), 3u);
    EXPECT_EQ(workload.collision_cells.size(), 3u);
    EXPECT_EQ(workload.filtered_collision_counts.size(), 0u);
    EXPECT_EQ(workload.flattened_collision_count, 3);
    EXPECT_EQ(workload.collision_cells[0], 0);
    EXPECT_EQ(workload.collision_cells[1], 0);
    EXPECT_EQ(workload.collision_cells[2], 2);
}

TEST(DsmcFlattenWorkload, BuildFiltersCellsByAllocatedSolver) {
    atlas::DsmcFlattenWorkload<float> workload;
    atlas::DeviceBuffer<int> counts(3);
    counts[0] = 2;
    counts[1] = 3;
    counts[2] = 4;

    atlas::DeviceBuffer<int> owners(3);
    owners[0] = 1;
    owners[1] = 2;
    owners[2] = 1;

    ASSERT_TRUE(workload.build(
        atlas::raw_pointer_cast(counts.data()),
        3,
        atlas::raw_pointer_cast(owners.data()),
        2));

    EXPECT_EQ(workload.filtered_collision_counts.size(), 3u);
    EXPECT_EQ(workload.collision_cells.size(), 3u);
    EXPECT_EQ(workload.flattened_collision_count, 3);
    EXPECT_EQ(workload.collision_cells[0], 1);
    EXPECT_EQ(workload.collision_cells[1], 1);
    EXPECT_EQ(workload.collision_cells[2], 1);
}
