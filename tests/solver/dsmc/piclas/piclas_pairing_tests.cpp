#include <atlas/solver/dsmc/piclas/piclas_pairing.h>

#include <testkit/testkit.h>

TEST(PiclasPairing, RejectsInvalidCollisionIndex) {
    int lhs = 7;
    int rhs = 9;

    atlas::PiclasPairing::select_pair_offsets(lhs, rhs, 2, 4, 1, 13);
    EXPECT_EQ(lhs, -1);
    EXPECT_EQ(rhs, -1);
}

TEST(PiclasPairing, SelectPairOffsetsReturnsDistinctOffsets) {
    int lhs = -1;
    int rhs = -1;

    atlas::PiclasPairing::select_pair_offsets(lhs, rhs, 1, 6, 4, 23);
    EXPECT_GE(lhs, 0);
    EXPECT_LT(lhs, 6);
    EXPECT_GE(rhs, 0);
    EXPECT_LT(rhs, 6);
    EXPECT_NE(lhs, rhs);
}
