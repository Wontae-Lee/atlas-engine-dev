#include "../utilities/test_utils.h"

#include <atlas/scheduler/dsmc_cell_sequential_scheduler.h>
#include <atlas/scheduler/dsmc_flatten_scheduler.h>
#include <atlas/scheduler/dsmc_piclas_scheduler.h>

#include <testkit/testkit.h>

namespace {

using DsmcCellSequentialScheduler = atlas::scheduler::DsmcCellSequentialScheduler<float>;
using DsmcFlattenScheduler        = atlas::scheduler::DsmcFlattenScheduler<float>;
using DsmcPiclasScheduler         = atlas::scheduler::DsmcPiclasScheduler<float>;
using DsmcProbe                   = atlas::system::DsmcProbe<float>;

}

TEST(DsmcCellSequentialScheduler, SelectPairOffsetsReturnsDistinctOffsets) {
    DsmcProbe probe;
    probe.collision_seed = 17;

    int lhs = -1;
    int rhs = -1;

    EXPECT_TRUE(DsmcCellSequentialScheduler::select_pair_offsets(probe, lhs, rhs, 3, 2, 8));
    EXPECT_GE(lhs, 0);
    EXPECT_LT(lhs, 8);
    EXPECT_GE(rhs, 0);
    EXPECT_LT(rhs, 8);
    EXPECT_NE(lhs, rhs);
}

TEST(DsmcFlattenScheduler, SelectPairOffsetsMatchesCellSequentialScheduler) {
    DsmcProbe probe;
    probe.collision_seed = 29;

    int cell_lhs = -1;
    int cell_rhs = -1;
    int flatten_lhs = -1;
    int flatten_rhs = -1;

    ASSERT_TRUE(DsmcCellSequentialScheduler::select_pair_offsets(probe, cell_lhs, cell_rhs, 5, 4, 11));
    ASSERT_TRUE(DsmcFlattenScheduler::select_pair_offsets(probe, flatten_lhs, flatten_rhs, 5, 4, 11));
    EXPECT_EQ(flatten_lhs, cell_lhs);
    EXPECT_EQ(flatten_rhs, cell_rhs);
}

TEST(DsmcPiclasScheduler, LimitsCollisionCount) {
    const DsmcPiclasScheduler scheduler;
    EXPECT_TRUE(scheduler.limits_collision_count());
}

TEST(DsmcPiclasScheduler, SelectPairOffsetsRejectsInvalidCollisionIndex) {
    int lhs = 7;
    int rhs = 9;

    DsmcPiclasScheduler::select_pair_offsets(lhs, rhs, 2, 4, 1, 13);
    EXPECT_EQ(lhs, -1);
    EXPECT_EQ(rhs, -1);
}

TEST(DsmcPiclasScheduler, SelectPairOffsetsReturnsDistinctOffsets) {
    int lhs = -1;
    int rhs = -1;

    DsmcPiclasScheduler::select_pair_offsets(lhs, rhs, 1, 6, 4, 23);
    EXPECT_GE(lhs, 0);
    EXPECT_LT(lhs, 6);
    EXPECT_GE(rhs, 0);
    EXPECT_LT(rhs, 6);
    EXPECT_NE(lhs, rhs);
}
