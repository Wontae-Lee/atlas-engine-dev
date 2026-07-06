#include <atlas/math/vector/float3.h>

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

constexpr float k_eps = 1e-6f;

}

TEST(Vector3, DefaultConstructibleToZero) {
    const atlas::Float3 v {};
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST(Vector3, UniformScalarConstructor) {
    const atlas::Float3 v(2.5f);
    EXPECT_FLOAT_EQ(v.x, 2.5f);
    EXPECT_FLOAT_EQ(v.y, 2.5f);
    EXPECT_FLOAT_EQ(v.z, 2.5f);
}

TEST(Vector3, ComponentConstructor) {
    const atlas::Float3 v(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, InitializerListConstructor) {
    const atlas::Float3 v { 1.0f, 2.0f, 3.0f };
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, DataAndIndexAccess) {
    atlas::Float3 v(1.0f, 2.0f, 3.0f);

    const float* p = v.data();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);

    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);

    v[0] = 10.0f;
    v[1] = 20.0f;
    v[2] = 30.0f;

    EXPECT_FLOAT_EQ(v.x, 10.0f);
    EXPECT_FLOAT_EQ(v.y, 20.0f);
    EXPECT_FLOAT_EQ(v.z, 30.0f);
}

TEST(Vector3, SetAndSetZero) {
    atlas::Float3 v(1.0f, 2.0f, 3.0f);

    v.set(7.0f);
    EXPECT_FLOAT_EQ(v.x, 7.0f);
    EXPECT_FLOAT_EQ(v.y, 7.0f);
    EXPECT_FLOAT_EQ(v.z, 7.0f);

    v.set_zero();
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);

    v.set_values(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, ScalarOpsInPlace) {
    atlas::Float3 v(1.0f, 2.0f, 3.0f);

    v.add(1.0f);
    EXPECT_FLOAT_EQ(v.y, 3.0f);

    v.sub(2.0f);
    EXPECT_FLOAT_EQ(v.y, 1.0f);

    v.mul(3.0f);
    EXPECT_FLOAT_EQ(v.y, 3.0f);

    v.div(2.0f);
    EXPECT_FLOAT_EQ(v.y, 1.5f);
}

TEST(Vector3, VectorOpsInPlace) {
    atlas::Float3 a(1.0f, 2.0f, 3.0f);
    const atlas::Float3 b(10.0f, 20.0f, 30.0f);

    a.add(b);
    EXPECT_FLOAT_EQ(a.x, 11.0f);

    a.sub(atlas::Float3(1.0f, 2.0f, 3.0f));
    EXPECT_FLOAT_EQ(a.x, 10.0f);

    a.mul(atlas::Float3(2.0f, 3.0f, 4.0f));
    EXPECT_FLOAT_EQ(a.z, 120.0f);

    a.div(atlas::Float3(2.0f, 3.0f, 4.0f));
    EXPECT_FLOAT_EQ(a.z, 30.0f);
}

TEST(Vector3, CompoundOperators) {
    atlas::Float3 v(1.0f, 2.0f, 3.0f);

    v += atlas::Float3(1.0f, 1.0f, 1.0f);
    v -= 1.0f;
    v *= 2.0f;
    v /= atlas::Float3(2.0f, 2.0f, 2.0f);

    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, EqualityIsExact) {
    const atlas::Float3 a(1.0f, 2.0f, 3.0f);
    const atlas::Float3 b(1.0f, 2.0f, 3.0f);
    const atlas::Float3 c(1.0f, 2.0f, 3.5f);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
}

TEST(Vector3, MinMax) {
    const atlas::Float3 v(-1.0f, 3.0f, 2.0f);
    EXPECT_FLOAT_EQ(v.min(), -1.0f);
    EXPECT_FLOAT_EQ(v.max(), 3.0f);
}

TEST(Vector3, DotAndCross) {
    const atlas::Float3 a(1.0f, 2.0f, 3.0f);
    const atlas::Float3 b(4.0f, 5.0f, 6.0f);

    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
    EXPECT_FLOAT_EQ(atlas::dot(a, b), 32.0f);

    const atlas::Float3 c = a.cross(b);
    EXPECT_FLOAT_EQ(c.x, -3.0f);
    EXPECT_FLOAT_EQ(c.y, 6.0f);
    EXPECT_FLOAT_EQ(c.z, -3.0f);

    const atlas::Float3 c2 = atlas::cross(a, b);
    EXPECT_FLOAT_EQ(c2.x, -3.0f);
    EXPECT_FLOAT_EQ(c2.y, 6.0f);
    EXPECT_FLOAT_EQ(c2.z, -3.0f);
}

TEST(Vector3, LengthAndNormalize) {
    atlas::Float3 v(3.0f, 4.0f, 12.0f);

    EXPECT_FLOAT_EQ(v.length_squared(), 169.0f);
    EXPECT_FLOAT_EQ(v.length(), 13.0f);

    const atlas::Float3 n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0f, k_eps);

    v.normalize();
    EXPECT_NEAR(v.length(), 1.0f, k_eps);
}

TEST(Vector3, NormalizeZeroVectorIsNoOp) {
    atlas::Float3 v {};
    v.normalize();
    EXPECT_FLOAT_EQ(v.length(), 0.0f);
}

TEST(Vector3, MajorMinorAxisDefinition) {
    const atlas::Float3 v(1.0f, -5.0f, 3.0f);

    const std::size_t maj = v.major_axis();
    const std::size_t min = v.minor_axis();

    ASSERT_LT(maj, static_cast<std::size_t>(3));
    ASSERT_LT(min, static_cast<std::size_t>(3));

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_GE(v[maj], v[i]);
        EXPECT_LE(v[min], v[i]);
    }
}

TEST(Vector3, ReflectedAndProjected) {
    const atlas::Float3 v(1.0f, -2.0f, 3.0f);
    const atlas::Float3 n(0.0f, 1.0f, 0.0f);

    const atlas::Float3 r = v.reflected(n);
    EXPECT_NEAR(r.x, 1.0f, k_eps);
    EXPECT_NEAR(r.y, 2.0f, k_eps);
    EXPECT_NEAR(r.z, 3.0f, k_eps);

    const atlas::Float3 p = v.projected(n);
    EXPECT_NEAR(p.y, 0.0f, k_eps);

    const atlas::Float3 r2 = atlas::reflected(v, n);
    EXPECT_NEAR(r2.y, 2.0f, k_eps);

    const atlas::Float3 p2 = atlas::projected(v, n);
    EXPECT_NEAR(p2.y, 0.0f, k_eps);
}

TEST(Vector3, TangentialBasisOrthogonality) {
    const atlas::Float3 n(0.0f, 0.0f, 1.0f);

    const auto [t0, t1] = n.tangential();

    EXPECT_NEAR(n.dot(t0), 0.0f, k_eps);
    EXPECT_NEAR(n.dot(t1), 0.0f, k_eps);
    EXPECT_NEAR(t0.dot(t1), 0.0f, k_eps);

    EXPECT_GT(t0.length(), 0.0f);
    EXPECT_GT(t1.length(), 0.0f);
}

TEST(Vector3, FreeOperatorsBasic) {
    const atlas::Float3 a(1.0f, 2.0f, 3.0f);
    const atlas::Float3 b(10.0f, 20.0f, 30.0f);

    EXPECT_TRUE((a + b) == atlas::Float3(11.0f, 22.0f, 33.0f));
    EXPECT_TRUE((a - b) == atlas::Float3(-9.0f, -18.0f, -27.0f));
    EXPECT_TRUE((a * 2.0f) == atlas::Float3(2.0f, 4.0f, 6.0f));
    EXPECT_TRUE((2.0f * a) == atlas::Float3(2.0f, 4.0f, 6.0f));
    EXPECT_TRUE((a * b) == atlas::Float3(10.0f, 40.0f, 90.0f));
    EXPECT_TRUE((-a) == atlas::Float3(-1.0f, -2.0f, -3.0f));

    const atlas::Float3 q = b / 10.0f;
    EXPECT_NEAR(q.x, 1.0f, k_eps);
    EXPECT_NEAR(q.y, 2.0f, k_eps);
    EXPECT_NEAR(q.z, 3.0f, k_eps);

    EXPECT_TRUE(atlas::cmin(a, b) == a);
    EXPECT_TRUE(atlas::cmax(a, b) == b);
    EXPECT_TRUE(atlas::min(a, b) == a);
    EXPECT_TRUE(atlas::max(a, b) == b);
}

TEST(Vector3, ClampFloorCeilAbs) {
    const atlas::Float3 v(-1.5f, 0.5f, 2.5f);

    const atlas::Float3 c = atlas::clamp(v, atlas::Float3(0.0f), atlas::Float3(1.0f));
    EXPECT_TRUE(c == atlas::Float3(0.0f, 0.5f, 1.0f));

    EXPECT_TRUE(atlas::floor(v) == atlas::Float3(-2.0f, 0.0f, 2.0f));
    EXPECT_TRUE(atlas::ceil(v) == atlas::Float3(-1.0f, 1.0f, 3.0f));
    EXPECT_TRUE(atlas::abs(v) == atlas::Float3(1.5f, 0.5f, 2.5f));
}

TEST(Vector3, RelationalOperatorsWithBool3) {
    const atlas::Float3 lo(0.0f, 0.0f, 0.0f);
    const atlas::Float3 hi(1.0f, 1.0f, 1.0f);
    const atlas::Float3 inside(0.5f, 0.5f, 0.5f);
    const atlas::Float3 outside(0.5f, 2.0f, 0.5f);

    EXPECT_TRUE(atlas::all(inside >= lo));
    EXPECT_TRUE(atlas::all(inside <= hi));
    EXPECT_FALSE(atlas::all(outside <= hi));
    EXPECT_TRUE(atlas::any(outside > hi));
    EXPECT_TRUE(atlas::none(lo > hi));
}

TEST(Vector3, IsFinite) {
    EXPECT_TRUE(atlas::isfinite(atlas::Float3(1.0f, 2.0f, 3.0f)));
    EXPECT_FALSE(atlas::isfinite(atlas::Float3(1.0f, atlas::inf, 3.0f)));
}

TEST(Vector3, NormalizedOrFallback) {
    const atlas::Float3 fallback(1.0f, 0.0f, 0.0f);

    const atlas::Float3 n = atlas::normalized_or(atlas::Float3(0.0f, 3.0f, 0.0f), fallback);
    EXPECT_NEAR(n.y, 1.0f, k_eps);

    const atlas::Float3 f = atlas::normalized_or(atlas::Float3 {}, fallback);
    EXPECT_TRUE(f == fallback);
}

TEST(Vector3, OrthonormalBasis) {
    const atlas::Float3 normal(0.0f, 0.0f, 2.0f);

    atlas::Float3 unit_normal;
    atlas::Float3 tangent;
    atlas::Float3 bitangent;
    ASSERT_TRUE(atlas::orthonormal_basis(normal, unit_normal, tangent, bitangent));

    EXPECT_NEAR(unit_normal.length(), 1.0f, k_eps);
    EXPECT_NEAR(tangent.length(), 1.0f, k_eps);
    EXPECT_NEAR(bitangent.length(), 1.0f, k_eps);
    EXPECT_NEAR(unit_normal.dot(tangent), 0.0f, k_eps);
    EXPECT_NEAR(unit_normal.dot(bitangent), 0.0f, k_eps);
    EXPECT_NEAR(tangent.dot(bitangent), 0.0f, k_eps);

    EXPECT_FALSE(atlas::orthonormal_basis(atlas::Float3 {}, unit_normal, tangent, bitangent));
}

TEST(Vector3, SphericalDirectionIsUnit) {
    const atlas::Float3 axis(0.0f, 0.0f, 1.0f);

    const atlas::Float3 d1 = atlas::spherical_direction(axis, 0.5f, 1.2f);
    EXPECT_NEAR(d1.length(), 1.0f, k_eps);

    const atlas::Float3 d2 = atlas::spherical_direction(0.5f, 1.2f);
    EXPECT_NEAR(d2.length(), 1.0f, k_eps);
    EXPECT_NEAR(d2.z, 0.5f, k_eps);
}

TEST(Vector3, XyHelpers) {
    const atlas::Float3 a(3.0f, 4.0f, 100.0f);

    EXPECT_FLOAT_EQ(atlas::xy_length_squared(a), 25.0f);
    EXPECT_FLOAT_EQ(atlas::xy_length(a), 5.0f);

    const atlas::Float3 n = atlas::xy_normalized_or(a, atlas::Float3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(n.x, 0.6f, k_eps);
    EXPECT_NEAR(n.y, 0.8f, k_eps);
    EXPECT_FLOAT_EQ(n.z, 0.0f);
}
