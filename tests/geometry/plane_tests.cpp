#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/plane.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Plane, DefaultConstructorCreatesValidPlane) {
    const atlas::geometry::Plane<T> plane;

    EXPECT_EQ(plane.type(), atlas::geometry::GeometryType::Plane);
    EXPECT_TRUE(plane.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(plane.normal, Vec3(0, 0, 1), kEps));
    EXPECT_NEAR(plane.offset, 0.0f, kEps);
}

TEST(Plane, BuilderConstructsConfiguredPlane) {
    const auto plane = atlas::geometry::Plane<T>::builder()
                           .with_point_normal(Vec3(0, 2, 0), Vec3(0, 1, 0))
                           .build();

    EXPECT_TRUE(atlas::test::vec_near(plane.normal, Vec3(0, 1, 0), kEps));
    EXPECT_NEAR(plane.offset, -2.0f, kEps);
}

TEST(Plane, BuilderRejectsDegenerateNormal) {
    EXPECT_THROW(
        atlas::geometry::Plane<T>::builder()
            .with_normal(Vec3(0, 0, 0))
            .build(),
        std::runtime_error);
}

TEST(Plane, ClosestPointNormalAndDistanceWork) {
    const atlas::geometry::Plane<T> plane(Vec3(0, 1, 0), -2.0f);

    const Vec3 closest = plane.closest_point(Vec3(1, 5, 3));
    const Vec3 normal = plane.closest_normal(Vec3(1, 5, 3));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(normal, Vec3(0, 1, 0), kEps));
    EXPECT_NEAR(plane.signed_distance(Vec3(1, 5, 3)), 3.0f, kEps);
}

TEST(Plane, ClassificationCentroidBoundAndTraceWork) {
    const atlas::geometry::Plane<T> plane(Vec3(0, 1, 0), -2.0f);

    EXPECT_TRUE(plane.is_inside(Vec3(0, 1, 0), 1.0f));
    EXPECT_TRUE(plane.is_on_surface(Vec3(0, 2, 0), kEps));

    const Vec3 center = plane.centroid();
    const auto bounds = plane.bound();
    const auto geometry_operator = plane.make_geometry_operator();
    const auto hit = plane.make_geometry_operator().trace(atlas::spatial::Ray<T>(Vec3(0, 5, 0), Vec3(0, -1, 0)));

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(0, 0, 0), kEps));
    EXPECT_EQ(plane.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 3.0f, kEps);
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(0, 2, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(hit.normal, Vec3(0, 1, 0), kEps));
    EXPECT_TRUE(bounds.is_valid());
}
