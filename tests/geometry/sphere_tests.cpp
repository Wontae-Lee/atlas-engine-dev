#include <atlas/geometry/sphere.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::GeometryType;
using atlas::Ray;
using atlas::Sphere;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Sphere, DefaultConstructorCreatesValidSphere) {
    const Sphere sphere;

    const atlas::Geometry geometry_operator(sphere);
    EXPECT_EQ(geometry_operator.type, GeometryType::sphere);
    EXPECT_TRUE(sphere.is_valid());
    EXPECT_NEAR(sphere.radius, 1.0f, tol);
}

TEST(Sphere, BuilderConstructsConfiguredSphere) {
    const auto sphere = Sphere::builder()
                            .with_center(Float3(1.0f, 2.0f, 3.0f))
                            .with_radius(4.0f)
                            .build();

    expect_vec_near(sphere.center, Float3(1.0f, 2.0f, 3.0f));
    EXPECT_NEAR(sphere.radius, 4.0f, tol);
}

TEST(Sphere, BuilderRejectsInvalidSphere) {
    EXPECT_THROW(
        (void)Sphere::builder()
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Sphere, ClosestPointNormalDistanceAndClassificationWork) {
    const Sphere sphere(Float3(0.0f, 0.0f, 0.0f), 2.0f);

    expect_vec_near(sphere.closest_point(Float3(4.0f, 0.0f, 0.0f)), Float3(2.0f, 0.0f, 0.0f));
    expect_vec_near(sphere.closest_normal(Float3(4.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    EXPECT_LT(sphere.signed_distance(Float3(0.0f, 0.0f, 0.0f)), 0.0f);
    EXPECT_NEAR(sphere.signed_distance(Float3(2.0f, 0.0f, 0.0f)), 0.0f, tol);
    EXPECT_TRUE(sphere.is_inside(Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(sphere.is_on_surface(Float3(2.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Sphere, CentroidBoundOperatorAndTraceWork) {
    const Sphere sphere(Float3(1.0f, 2.0f, 3.0f), 2.0f);

    const Float3 center            = sphere.centroid();
    const auto bounds               = sphere.bound();
    const atlas::Geometry geometry_operator(sphere);
    const auto hit = geometry_operator.trace(Ray(Float3(5.0f, 2.0f, 3.0f), Float3(-1.0f, 0.0f, 0.0f)));

    expect_vec_near(center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(bounds.lower_corner, Float3(-1.0f, 0.0f, 1.0f));
    expect_vec_near(bounds.upper_corner, Float3(3.0f, 4.0f, 5.0f));
    EXPECT_EQ(GeometryType::sphere, geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
}
