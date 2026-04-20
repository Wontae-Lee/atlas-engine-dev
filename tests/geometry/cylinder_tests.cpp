#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/cylinder.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Cylinder, DefaultConstructorCreatesValidCylinder) {
    const atlas::geometry::Cylinder<T> cylinder;

    EXPECT_EQ(cylinder.type(), atlas::geometry::GeometryType::Cylinder);
    EXPECT_TRUE(cylinder.is_valid());
    EXPECT_NEAR(cylinder.radius, 1.0f, kEps);
    EXPECT_NEAR(cylinder.height, 1.0f, kEps);
}

TEST(Cylinder, BuilderConstructsConfiguredCylinder) {
    const auto cylinder = atlas::geometry::Cylinder<T>::builder()
                              .with_center(Vec3(1, 2, 3))
                              .with_radius(2.5f)
                              .with_height(6.0f)

                              .build();

    EXPECT_TRUE(atlas::test::vec_near(cylinder.center, Vec3(1, 2, 3), kEps));
    EXPECT_NEAR(cylinder.radius, 2.5f, kEps);
    EXPECT_NEAR(cylinder.height, 6.0f, kEps);
}

TEST(Cylinder, BuilderRejectsInvalidCylinder) {
    EXPECT_THROW(
        atlas::geometry::Cylinder<T>::builder()
            .with_radius(-1.0f)
            .with_height(0.0f)
            .build(),
        std::runtime_error);
}

TEST(Cylinder, ClosestPointAndNormalWork) {
    const atlas::geometry::Cylinder<T> cylinder(Vec3(0, 0, 0), 2.0f, 4.0f);

    const Vec3 closest = cylinder.closest_point(Vec3(4, 0, 0));
    const Vec3 normal = cylinder.closest_normal(Vec3(4, 0, 0));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(2, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(normal, Vec3(1, 0, 0), kEps));
}

TEST(Cylinder, SignedDistanceAndClassificationWork) {
    const atlas::geometry::Cylinder<T> cylinder(Vec3(0, 0, 0), 2.0f, 4.0f);

    EXPECT_LT(cylinder.signed_distance(Vec3(0, 0, 0)), 0.0f);
    EXPECT_NEAR(cylinder.signed_distance(Vec3(2, 0, 0)), 0.0f, kEps);
    EXPECT_GT(cylinder.signed_distance(Vec3(4, 0, 0)), 0.0f);

    EXPECT_TRUE(cylinder.is_inside(Vec3(0, 0, 0), 0.0f));
    EXPECT_TRUE(cylinder.is_on_surface(Vec3(2, 0, 0), 0.0f));
}

TEST(Cylinder, CentroidBoundAndOperatorWork) {
    const atlas::geometry::Cylinder<T> cylinder(Vec3(1, 2, 3), 2.0f, 4.0f);

    const Vec3 center = cylinder.centroid();
    const auto bounds = cylinder.bound();
    const auto geometry_operator = cylinder.make_geometry_operator();

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.lower_corner, Vec3(-1, 0, 1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.upper_corner, Vec3(3, 4, 5), kEps));
    EXPECT_EQ(cylinder.type(), geometry_operator.type);
}
