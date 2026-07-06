#include <atlas/geometry/plane.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::GeometryType;
using atlas::Plane;
using atlas::Ray;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Plane, DefaultConstructorCreatesValidPlane) {
    const Plane plane;

    const atlas::Geometry geometry_operator(plane);
    EXPECT_EQ(geometry_operator.type, GeometryType::plane);
    EXPECT_TRUE(plane.is_valid());
    expect_vec_near(plane.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(plane.offset, 0.0f, tol);
}

TEST(Plane, BuilderConstructsConfiguredPlane) {
    const auto plane = Plane::builder()
                           .with_point_normal(Float3(0.0f, 2.0f, 0.0f), Float3(0.0f, 1.0f, 0.0f))
                           .build();

    expect_vec_near(plane.normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_NEAR(plane.offset, -2.0f, tol);
}

TEST(Plane, BuilderRejectsDegenerateNormal) {
    EXPECT_THROW(
        (void)Plane::builder()
            .with_normal(Float3(0.0f, 0.0f, 0.0f))
            .build(),
        std::runtime_error);
}

TEST(Plane, ClosestPointNormalAndDistanceWork) {
    const Plane plane(Float3(0.0f, 1.0f, 0.0f), -2.0f);

    const Float3 closest = plane.closest_point(Float3(1.0f, 5.0f, 3.0f));
    const Float3 normal  = plane.closest_normal(Float3(1.0f, 5.0f, 3.0f));

    expect_vec_near(closest, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_NEAR(plane.signed_distance(Float3(1.0f, 5.0f, 3.0f)), 3.0f, tol);
}

TEST(Plane, ClassificationCentroidBoundAndTraceWork) {
    const Plane plane(Float3(0.0f, 1.0f, 0.0f), -2.0f);

    EXPECT_TRUE(plane.is_inside(Float3(0.0f, 1.0f, 0.0f), 1.0f));
    EXPECT_TRUE(plane.is_on_surface(Float3(0.0f, 2.0f, 0.0f), tol));

    const Float3 center            = plane.centroid();
    const auto bounds               = plane.bound();
    const atlas::Geometry geometry_operator(plane);
    const auto hit = geometry_operator.trace(Ray(Float3(0.0f, 5.0f, 0.0f), Float3(0.0f, -1.0f, 0.0f)));

    expect_vec_near(center, Float3(0.0f, 0.0f, 0.0f));
    EXPECT_EQ(GeometryType::plane, geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 3.0f, tol);
    expect_vec_near(hit.point, Float3(0.0f, 2.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_TRUE(bounds.is_valid());
}
