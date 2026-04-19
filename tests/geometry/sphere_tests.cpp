#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/sphere.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Sphere, DefaultConstructorCreatesValidSphere) {
    const atlas::geometry::Sphere<T> sphere;

    EXPECT_EQ(sphere.type(), atlas::geometry::GeometryType::Sphere);
    EXPECT_TRUE(sphere.is_valid());
    EXPECT_NEAR(sphere.radius, 1.0f, kEps);
}

TEST(Sphere, BuilderConstructsConfiguredSphere) {
    const auto sphere = atlas::geometry::Sphere<T>::builder()
                            .with_center(Vec3(1, 2, 3))
                            .with_radius(4.0f)
                            .build();

    EXPECT_TRUE(atlas::test::vec_near(sphere.center, Vec3(1, 2, 3), kEps));
    EXPECT_NEAR(sphere.radius, 4.0f, kEps);
}

TEST(Sphere, BuilderRejectsInvalidSphere) {
    EXPECT_THROW(
        atlas::geometry::Sphere<T>::builder()
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Sphere, ClosestPointNormalDistanceAndClassificationWork) {
    const atlas::geometry::Sphere<T> sphere(Vec3(0, 0, 0), 2.0f);

    EXPECT_TRUE(atlas::test::vec_near(sphere.closest_point(Vec3(4, 0, 0)), Vec3(2, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(sphere.closest_normal(Vec3(4, 0, 0)), Vec3(1, 0, 0), kEps));
    EXPECT_LT(sphere.signed_distance(Vec3(0, 0, 0)), 0.0f);
    EXPECT_NEAR(sphere.signed_distance(Vec3(2, 0, 0)), 0.0f, kEps);
    EXPECT_TRUE(sphere.is_inside(Vec3(0, 0, 0), 0.0f));
    EXPECT_TRUE(sphere.is_on_surface(Vec3(2, 0, 0), 0.0f));
}

TEST(Sphere, CentroidBoundOperatorAndTraceWork) {
    const atlas::geometry::Sphere<T> sphere(Vec3(1, 2, 3), 2.0f);

    const Vec3 center = sphere.centroid();
    const auto bounds = sphere.bound();
    const auto geometry_operator = sphere.make_geometry_operator();
    const auto hit = sphere.make_geometry_operator().trace(atlas::spatial::Ray<T>(Vec3(5, 2, 3), Vec3(-1, 0, 0)));

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.lower_corner, Vec3(-1, 0, 1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.upper_corner, Vec3(3, 4, 5), kEps));
    EXPECT_EQ(sphere.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
}
