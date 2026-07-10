#include <atlas/math/vector/int3.h>

#include <gtest/gtest.h>

namespace {

using atlas::Bool3;
using atlas::Float3;
using atlas::Int3;

}

TEST(Int3, DefaultConstructsToZero) {
    const Int3 v;
    EXPECT_EQ(v.x, 0);
    EXPECT_EQ(v.y, 0);
    EXPECT_EQ(v.z, 0);
}

TEST(Int3, ScalarConstructorFillsEveryComponent) {
    EXPECT_TRUE(Int3(4) == Int3(4, 4, 4));
}

TEST(Int3, IndexingMatchesNamedComponents) {
    const Int3 v(1, 2, 3);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
    EXPECT_EQ(v.data(), &v.x);
}

TEST(Int3, EqualityIsExactComponentWise) {
    EXPECT_TRUE(Int3(1, 2, 3) == Int3(1, 2, 3));
    EXPECT_TRUE(Int3(1, 2, 3) != Int3(1, 2, 4));
}

TEST(Int3, AdditionAndSubtractionAreInverse) {
    const Int3 a(1, -2, 3);
    const Int3 b(4, 5, -6);
    EXPECT_TRUE((a + b) - b == a);
}

TEST(Int3, UnaryNegationFlipsEverySign) {
    EXPECT_TRUE(-Int3(1, -2, 3) == Int3(-1, 2, -3));
}

TEST(Int3, ScalarMultiplicationCommutes) {
    const Int3 v(1, 2, 3);
    EXPECT_TRUE(3 * v == v * 3);
    EXPECT_TRUE(v * 3 == Int3(3, 6, 9));
}

TEST(Int3, CompoundAssignmentMatchesBinaryForm) {
    Int3 v(1, 2, 3);
    v += Int3(1, 1, 1);
    v -= Int3(0, 1, 0);
    v *= 2;
    EXPECT_TRUE(v == Int3(4, 4, 8));
}

TEST(Int3, MinAndMaxSelectTheExtremeComponent) {
    const Int3 v(-2, 5, 1);
    EXPECT_EQ(v.min(), -2);
    EXPECT_EQ(v.max(), 5);
}

TEST(Int3, ComponentWiseMinMaxTakePerAxisExtremes) {
    const Int3 a(1, 5, 3);
    const Int3 b(4, 2, 6);
    EXPECT_TRUE(atlas::min(a, b) == Int3(1, 2, 3));
    EXPECT_TRUE(atlas::max(a, b) == Int3(4, 5, 6));
}

TEST(Int3, ClampConstrainsEachAxisToTheBox) {
    const Int3 v(-1, 5, 2);
    EXPECT_TRUE(atlas::clamp(v, Int3(0), Int3(3)) == Int3(0, 3, 2));
}

TEST(Int3, RelationalOperatorsReturnPerAxisBool3) {
    const Int3 a(1, 5, 3);
    const Int3 b(2, 2, 3);
    const Bool3 le = a <= b;
    EXPECT_TRUE(le.x);
    EXPECT_FALSE(le.y);
    EXPECT_TRUE(le.z);
    EXPECT_TRUE(atlas::any(a > b));
}

TEST(Int3, ToVectorTruncatesTowardZero) {
    // static_cast<int> truncates toward zero, so -1.9 becomes -1, not -2.
    EXPECT_TRUE(atlas::to_vector3i(Float3(1.9f, -1.9f, 2.5f)) == Int3(1, -1, 2));
}

TEST(Int3, ToFloatWidensEachComponent) {
    const Float3 v = atlas::to_vector3(Int3(1, -2, 3));
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}
