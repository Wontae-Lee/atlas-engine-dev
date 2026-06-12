#include "../utilities/test_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/sphere.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Ray;
using atlas::Sphere;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Sphere, DefaultConstructorCreatesValidSphere) {
    const Sphere<float> sphere;

    EXPECT_EQ(sphere.type(), GeometryType::Sphere);
    EXPECT_TRUE(sphere.is_valid());
    EXPECT_NEAR(sphere.radius, 1.0f, tol);
}

TEST(Sphere, BuilderConstructsConfiguredSphere) {
    const auto sphere = Sphere<float>::builder()
                            .with_center(Vector3F(1, 2, 3))
                            .with_radius(4.0f)
                            .build();

    EXPECT_TRUE(vec_near(sphere.center, Vector3F(1, 2, 3), tol));
    EXPECT_NEAR(sphere.radius, 4.0f, tol);
}

TEST(Sphere, BuilderRejectsInvalidSphere) {
    EXPECT_THROW(
        Sphere<float>::builder()
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Sphere, ClosestPointNormalDistanceAndClassificationWork) {
    const Sphere<float> sphere(Vector3F(0, 0, 0), 2.0f);

    EXPECT_TRUE(vec_near(sphere.closest_point(Vector3F(4, 0, 0)), Vector3F(2, 0, 0), tol));
    EXPECT_TRUE(vec_near(sphere.closest_normal(Vector3F(4, 0, 0)), Vector3F(1, 0, 0), tol));
    EXPECT_LT(sphere.signed_distance(Vector3F(0, 0, 0)), 0.0f);
    EXPECT_NEAR(sphere.signed_distance(Vector3F(2, 0, 0)), 0.0f, tol);
    EXPECT_TRUE(sphere.is_inside(Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(sphere.is_on_surface(Vector3F(2, 0, 0), 0.0f));
}

TEST(Sphere, CentroidBoundOperatorAndTraceWork) {
    const Sphere<float> sphere(Vector3F(1, 2, 3), 2.0f);

    const Vector3F center = sphere.centroid();
    const auto bounds = sphere.bound();
    const auto geometry_operator = sphere.make_device_geometry_view();
    const auto hit = sphere.make_device_geometry_view().trace(Ray<float>(Vector3F(5, 2, 3), Vector3F(-1, 0, 0)));

    EXPECT_TRUE(vec_near(center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(bounds.lower_corner, Vector3F(-1, 0, 1), tol));
    EXPECT_TRUE(vec_near(bounds.upper_corner, Vector3F(3, 4, 5), tol));
    EXPECT_EQ(sphere.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
}
