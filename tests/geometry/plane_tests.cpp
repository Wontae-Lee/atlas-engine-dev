#include "../utilities/test_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/plane.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Plane;
using atlas::Ray;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Plane, DefaultConstructorCreatesValidPlane) {
    const Plane<float> plane;

    EXPECT_EQ(plane.type(), GeometryType::Plane);
    EXPECT_TRUE(plane.is_valid());
    EXPECT_TRUE(vec_near(plane.normal, Vector3F(0, 0, 1), tol));
    EXPECT_NEAR(plane.offset, 0.0f, tol);
}

TEST(Plane, BuilderConstructsConfiguredPlane) {
    const auto plane = Plane<float>::builder()
                           .with_point_normal(Vector3F(0, 2, 0), Vector3F(0, 1, 0))
                           .build();

    EXPECT_TRUE(vec_near(plane.normal, Vector3F(0, 1, 0), tol));
    EXPECT_NEAR(plane.offset, -2.0f, tol);
}

TEST(Plane, BuilderRejectsDegenerateNormal) {
    EXPECT_THROW(
        Plane<float>::builder()
            .with_normal(Vector3F(0, 0, 0))
            .build(),
        std::runtime_error);
}

TEST(Plane, ClosestPointNormalAndDistanceWork) {
    const Plane<float> plane(Vector3F(0, 1, 0), -2.0f);

    const Vector3F closest = plane.closest_point(Vector3F(1, 5, 3));
    const Vector3F normal = plane.closest_normal(Vector3F(1, 5, 3));

    EXPECT_TRUE(vec_near(closest, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(normal, Vector3F(0, 1, 0), tol));
    EXPECT_NEAR(plane.signed_distance(Vector3F(1, 5, 3)), 3.0f, tol);
}

TEST(Plane, ClassificationCentroidBoundAndTraceWork) {
    const Plane<float> plane(Vector3F(0, 1, 0), -2.0f);

    EXPECT_TRUE(plane.is_inside(Vector3F(0, 1, 0), 1.0f));
    EXPECT_TRUE(plane.is_on_surface(Vector3F(0, 2, 0), tol));

    const Vector3F center = plane.centroid();
    const auto bounds = plane.bound();
    const auto geometry_operator = plane.make_device_geometry_view();
    const auto hit = plane.make_device_geometry_view().trace(Ray<float>(Vector3F(0, 5, 0), Vector3F(0, -1, 0)));

    EXPECT_TRUE(vec_near(center, Vector3F(0, 0, 0), tol));
    EXPECT_EQ(plane.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 3.0f, tol);
    EXPECT_TRUE(vec_near(hit.point, Vector3F(0, 2, 0), tol));
    EXPECT_TRUE(vec_near(hit.normal, Vector3F(0, 1, 0), tol));
    EXPECT_TRUE(bounds.is_valid());
}
