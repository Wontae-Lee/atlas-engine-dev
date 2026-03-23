#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(PlaneGeometryOperator, IsInsideUsesHalfSpaceAndTolerance) {
    const Vector3<double> n(0.0, 0.0, 1.0);
    constexpr double d = 0.0;

    geometry::PlaneGeometryOperator<double> op;
    op.normal = atlas::raw_pointer_cast(&n);
    op.offset = atlas::raw_pointer_cast(&d);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, -1.0)));
    EXPECT_FALSE(op.is_inside(Vector3<double>(0.0, 0.0, 0.2)));
    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, 0.2), 0.25));
}

TEST(PlaneGeometryOperator, IsOnSurfaceDetectsPlaneBand) {
    const Vector3<double> n(0.0, 0.0, 1.0);
    constexpr double d = 0.0;

    geometry::PlaneGeometryOperator<double> op;
    op.normal = atlas::raw_pointer_cast(&n);
    op.offset = atlas::raw_pointer_cast(&d);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 2.0, 0.0)));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(1.0, 2.0, 0.5)));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 2.0, 0.1), 0.15));
}