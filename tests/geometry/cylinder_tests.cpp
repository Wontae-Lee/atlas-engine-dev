#include "../utilities/test_utils.h"

#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Cylinder;
using atlas::Vector3F;
using atlas::geometry::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Cylinder, DefaultConstructorCreatesValidCylinder) {
    const Cylinder<float> cylinder;

    EXPECT_EQ(cylinder.type(), GeometryType::Cylinder);
    EXPECT_TRUE(cylinder.is_valid());
    EXPECT_NEAR(cylinder.radius, 1.0f, tol);
    EXPECT_NEAR(cylinder.height, 1.0f, tol);
}

TEST(Cylinder, BuilderConstructsConfiguredCylinder) {
    const auto cylinder = Cylinder<float>::builder()
                              .with_center(Vector3F(1, 2, 3))
                              .with_radius(2.5f)
                              .with_height(6.0f)

                              .build();

    EXPECT_TRUE(vec_near(cylinder.center, Vector3F(1, 2, 3), tol));
    EXPECT_NEAR(cylinder.radius, 2.5f, tol);
    EXPECT_NEAR(cylinder.height, 6.0f, tol);
}

TEST(Cylinder, BuilderRejectsInvalidCylinder) {
    EXPECT_THROW(
        Cylinder<float>::builder()
            .with_radius(-1.0f)
            .with_height(0.0f)
            .build(),
        std::runtime_error);
}

TEST(Cylinder, ClosestPointAndNormalWork) {
    const Cylinder<float> cylinder(Vector3F(0, 0, 0), 2.0f, 4.0f);

    const Vector3F closest = cylinder.closest_point(Vector3F(4, 0, 0));
    const Vector3F normal = cylinder.closest_normal(Vector3F(4, 0, 0));

    EXPECT_TRUE(vec_near(closest, Vector3F(2, 0, 0), tol));
    EXPECT_TRUE(vec_near(normal, Vector3F(1, 0, 0), tol));
}

TEST(Cylinder, SignedDistanceAndClassificationWork) {
    const Cylinder<float> cylinder(Vector3F(0, 0, 0), 2.0f, 4.0f);

    EXPECT_LT(cylinder.signed_distance(Vector3F(0, 0, 0)), 0.0f);
    EXPECT_NEAR(cylinder.signed_distance(Vector3F(2, 0, 0)), 0.0f, tol);
    EXPECT_GT(cylinder.signed_distance(Vector3F(4, 0, 0)), 0.0f);

    EXPECT_TRUE(cylinder.is_inside(Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(cylinder.is_on_surface(Vector3F(2, 0, 0), 0.0f));
}

TEST(Cylinder, CentroidBoundAndOperatorWork) {
    const Cylinder<float> cylinder(Vector3F(1, 2, 3), 2.0f, 4.0f);

    const Vector3F center = cylinder.centroid();
    const auto bounds = cylinder.bound();
    const auto geometry_operator = cylinder.make_geometry_operator();

    EXPECT_TRUE(vec_near(center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(bounds.lower_corner, Vector3F(-1, 0, 1), tol));
    EXPECT_TRUE(vec_near(bounds.upper_corner, Vector3F(3, 4, 5), tol));
    EXPECT_EQ(cylinder.type(), geometry_operator.type);
}
