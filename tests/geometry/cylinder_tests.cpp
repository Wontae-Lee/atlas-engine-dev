#include <atlas/geometry/cylinder.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Cylinder;
using atlas::GeometryType;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Cylinder, DefaultConstructorCreatesValidCylinder) {
    const Cylinder cylinder;

    const atlas::Geometry geometry_operator(cylinder);
    EXPECT_EQ(geometry_operator.type, GeometryType::cylinder);
    EXPECT_TRUE(cylinder.is_valid());
    EXPECT_NEAR(cylinder.radius, 1.0f, tol);
    EXPECT_NEAR(cylinder.height, 1.0f, tol);
}

TEST(Cylinder, BuilderConstructsConfiguredCylinder) {
    const auto cylinder = Cylinder::builder()
                              .with_center(Vector3(1.0f, 2.0f, 3.0f))
                              .with_radius(2.5f)
                              .with_height(6.0f)
                              .build();

    expect_vec_near(cylinder.center, Vector3(1.0f, 2.0f, 3.0f));
    EXPECT_NEAR(cylinder.radius, 2.5f, tol);
    EXPECT_NEAR(cylinder.height, 6.0f, tol);
}

TEST(Cylinder, BuilderRejectsInvalidCylinder) {
    EXPECT_THROW(
        (void)Cylinder::builder()
            .with_radius(-1.0f)
            .with_height(0.0f)
            .build(),
        std::runtime_error);
}

TEST(Cylinder, ClosestPointAndNormalWork) {
    const Cylinder cylinder(Vector3(0.0f, 0.0f, 0.0f), 2.0f, 4.0f);

    const Vector3 closest = cylinder.closest_point(Vector3(4.0f, 0.0f, 0.0f));
    const Vector3 normal  = cylinder.closest_normal(Vector3(4.0f, 0.0f, 0.0f));

    expect_vec_near(closest, Vector3(2.0f, 0.0f, 0.0f));
    expect_vec_near(normal, Vector3(1.0f, 0.0f, 0.0f));
}

TEST(Cylinder, SignedDistanceAndClassificationWork) {
    const Cylinder cylinder(Vector3(0.0f, 0.0f, 0.0f), 2.0f, 4.0f);

    EXPECT_LT(cylinder.signed_distance(Vector3(0.0f, 0.0f, 0.0f)), 0.0f);
    EXPECT_NEAR(cylinder.signed_distance(Vector3(2.0f, 0.0f, 0.0f)), 0.0f, tol);
    EXPECT_GT(cylinder.signed_distance(Vector3(4.0f, 0.0f, 0.0f)), 0.0f);

    EXPECT_TRUE(cylinder.is_inside(Vector3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(cylinder.is_on_surface(Vector3(2.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Cylinder, CentroidBoundAndOperatorWork) {
    const Cylinder cylinder(Vector3(1.0f, 2.0f, 3.0f), 2.0f, 4.0f);

    const Vector3 center            = cylinder.centroid();
    const auto bounds               = cylinder.bound();
    const atlas::Geometry geometry_operator(cylinder);

    expect_vec_near(center, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(bounds.lower_corner, Vector3(-1.0f, 0.0f, 1.0f));
    expect_vec_near(bounds.upper_corner, Vector3(3.0f, 4.0f, 5.0f));
    EXPECT_EQ(GeometryType::cylinder, geometry_operator.type);
}
