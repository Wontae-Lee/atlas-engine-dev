#include <atlas/math/vector/vector3i.h>

#include <gtest/gtest.h>

TEST(Vector3i, DefaultConstructibleToZero) {
    const atlas::Vector3i v {};
    EXPECT_EQ(v.x, 0);
    EXPECT_EQ(v.y, 0);
    EXPECT_EQ(v.z, 0);
}

TEST(Vector3i, Constructors) {
    const atlas::Vector3i s(7);
    EXPECT_EQ(s.x, 7);
    EXPECT_EQ(s.y, 7);
    EXPECT_EQ(s.z, 7);

    const atlas::Vector3i v(1, 2, 3);
    EXPECT_EQ(v.x, 1);
    EXPECT_EQ(v.y, 2);
    EXPECT_EQ(v.z, 3);
}

TEST(Vector3i, IndexAccess) {
    atlas::Vector3i v(1, 2, 3);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);

    v[2] = 30;
    EXPECT_EQ(v.z, 30);
}

TEST(Vector3i, Arithmetic) {
    const atlas::Vector3i a(1, 2, 3);
    const atlas::Vector3i b(10, 20, 30);

    EXPECT_TRUE((a + b) == atlas::Vector3i(11, 22, 33));
    EXPECT_TRUE((b - a) == atlas::Vector3i(9, 18, 27));
    EXPECT_TRUE((a * 2) == atlas::Vector3i(2, 4, 6));
    EXPECT_TRUE((2 * a) == atlas::Vector3i(2, 4, 6));
    EXPECT_TRUE((-a) == atlas::Vector3i(-1, -2, -3));

    atlas::Vector3i c = a;
    c += b;
    c -= a;
    c *= 2;
    EXPECT_TRUE(c == atlas::Vector3i(20, 40, 60));
}

TEST(Vector3i, MinMaxClamp) {
    const atlas::Vector3i v(-1, 5, 2);
    const atlas::Vector3i lo(0, 0, 0);
    const atlas::Vector3i hi(3, 3, 3);

    EXPECT_EQ(v.min(), -1);
    EXPECT_EQ(v.max(), 5);

    EXPECT_TRUE(atlas::min(v, lo) == atlas::Vector3i(-1, 0, 0));
    EXPECT_TRUE(atlas::max(v, lo) == atlas::Vector3i(0, 5, 2));
    EXPECT_TRUE(atlas::clamp(v, lo, hi) == atlas::Vector3i(0, 3, 2));
}

TEST(Vector3i, RelationalOperatorsWithBool3) {
    const atlas::Vector3i cell(1, 2, 3);
    const atlas::Vector3i zero(0, 0, 0);
    const atlas::Vector3i grid(4, 4, 4);

    EXPECT_TRUE(atlas::all(cell >= zero));
    EXPECT_TRUE(atlas::all(cell < grid));
    EXPECT_FALSE(atlas::all(cell < atlas::Vector3i(2, 2, 2)));
    EXPECT_TRUE(atlas::any(cell < atlas::Vector3i(2, 2, 2)));
}

TEST(Vector3i, ConversionWithVector3) {
    const atlas::Vector3 vf(1.9f, -2.1f, 3.5f);

    const atlas::Vector3i vi = atlas::to_vector3i(vf);
    EXPECT_EQ(vi.x, 1);
    EXPECT_EQ(vi.y, -2);
    EXPECT_EQ(vi.z, 3);

    const atlas::Vector3 back = atlas::to_vector3(vi);
    EXPECT_FLOAT_EQ(back.x, 1.0f);
    EXPECT_FLOAT_EQ(back.y, -2.0f);
    EXPECT_FLOAT_EQ(back.z, 3.0f);
}

TEST(Vector3i, FloorThenConvertMatchesGridSnap) {
    const atlas::Vector3 position(2.7f, -0.3f, 1.0f);
    const atlas::Vector3i cell = atlas::to_vector3i(atlas::floor(position));
    EXPECT_TRUE(cell == atlas::Vector3i(2, -1, 1));
}
