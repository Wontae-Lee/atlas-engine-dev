#include <atlas/unit/unit.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Box;
using atlas::Geometry;
using atlas::HitSurface;
using atlas::pi;
using atlas::Plane;
using atlas::Quaternion;
using atlas::Ray;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected, const float eps = tol) {
    EXPECT_NEAR(actual.x, expected.x, eps);
    EXPECT_NEAR(actual.y, expected.y, eps);
    EXPECT_NEAR(actual.z, expected.z, eps);
}

SyncHostPtr
make_sync(const Float3& translation, const Quaternion& orientation = Quaternion(1.0f, 0.0f, 0.0f, 0.0f)) {
    return Sync::builder().with_rigid_pose(translation, orientation).make_host_shared();
}

Geometry
make_box(const Float3& lower, const Float3& upper) {
    return Geometry(Box::builder().with_lower_corner(lower).with_upper_corner(upper).build());
}

Geometry
make_unit_box() {
    return make_box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));
}

// A local z = 0 plane facing +z.
Geometry
make_plane() {
    return Geometry(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f)));
}

}

TEST(Unit, BuilderRejectsMissingGeometry) {
    EXPECT_THROW(
        static_cast<void>(Unit::builder().with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f))).build()),
        std::runtime_error);
}

TEST(Unit, BuilderRejectsMissingSync) {
    EXPECT_THROW(
        static_cast<void>(Unit::builder().with_geometry(make_unit_box()).build()),
        std::runtime_error);
}

TEST(Unit, BuilderRejectsNullSync) {
    EXPECT_THROW(
        static_cast<void>(Unit::builder().with_sync(SyncHostPtr {})),
        std::runtime_error);
}

TEST(Unit, BuilderRejectsAccelerationWithoutVelocity) {
    EXPECT_THROW(
        static_cast<void>(Unit::builder()
                              .with_geometry(make_unit_box())
                              .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                              .with_acceleration(Float3(0.0f, 0.0f, 1.0f))
                              .build()),
        std::runtime_error);
}

TEST(Unit, BuilderRejectsAngularAccelerationWithoutAngularVelocity) {
    EXPECT_THROW(
        static_cast<void>(Unit::builder()
                              .with_geometry(make_unit_box())
                              .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                              .with_angular_acceleration(Float3(0.0f, 0.0f, 1.0f))
                              .build()),
        std::runtime_error);
}

TEST(Unit, BuilderBuildsStaticUnit) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .build();

    EXPECT_FALSE(unit.dynamic());
    EXPECT_FALSE(unit.velocity().has_value());
    EXPECT_FALSE(unit.acceleration().has_value());
    EXPECT_FALSE(unit.angular_velocity().has_value());
    EXPECT_FALSE(unit.angular_acceleration().has_value());
}

TEST(Unit, DefaultConstructedUnitIsStatic) {
    const Unit unit {};

    EXPECT_FALSE(unit.dynamic());
    EXPECT_FALSE(unit.velocity().has_value());
}

TEST(Unit, MakeHostSharedBuildsUnit) {
    const auto unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(unit));
    EXPECT_FALSE(unit->dynamic());
}

TEST(Unit, BuilderRoundTripsKinematics) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .with_velocity(Float3(1.0f, 2.0f, 3.0f))
                          .with_acceleration(Float3(4.0f, 5.0f, 6.0f))
                          .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
                          .with_angular_acceleration(Float3(0.0f, 0.0f, 2.0f))
                          .build();

    ASSERT_TRUE(unit.velocity().has_value());
    ASSERT_TRUE(unit.acceleration().has_value());
    ASSERT_TRUE(unit.angular_velocity().has_value());
    ASSERT_TRUE(unit.angular_acceleration().has_value());
    expect_vec_near(*unit.velocity(), Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(*unit.acceleration(), Float3(4.0f, 5.0f, 6.0f));
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(*unit.angular_acceleration(), Float3(0.0f, 0.0f, 2.0f));
}

TEST(Unit, BuilderCanonicalizesZeroPartner) {
    // A lone velocity gets a zero acceleration partner so integration is consistent.
    const Unit linear = Unit::builder()
                            .with_geometry(make_unit_box())
                            .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                            .with_velocity(Float3(1.0f, 0.0f, 0.0f))
                            .build();

    ASSERT_TRUE(linear.acceleration().has_value());
    expect_vec_near(*linear.acceleration(), Float3(0.0f, 0.0f, 0.0f));

    const Unit angular = Unit::builder()
                             .with_geometry(make_unit_box())
                             .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                             .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
                             .build();

    ASSERT_TRUE(angular.angular_acceleration().has_value());
    expect_vec_near(*angular.angular_acceleration(), Float3(0.0f, 0.0f, 0.0f));
}

TEST(Unit, DynamicReportsAnyMotion) {
    const Unit linear = Unit::builder()
                            .with_geometry(make_unit_box())
                            .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                            .with_velocity(Float3(0.0f, 0.0f, 1.0f))
                            .build();
    EXPECT_TRUE(linear.dynamic());

    const Unit angular = Unit::builder()
                             .with_geometry(make_unit_box())
                             .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                             .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
                             .build();
    EXPECT_TRUE(angular.dynamic());
}

TEST(Unit, UpdateMovesDynamicUnitByVelocityTimesDt) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .with_velocity(Float3(0.0f, 0.0f, 2.0f))
                    .build();

    unit.update(0.5f);

    // v (2) * dt (0.5) = 1.0 along z, with no acceleration to add.
    EXPECT_NEAR(unit.sync().translation.z, 1.0f, tol);
}

TEST(Unit, UpdateIntegratesAccelerationSemiImplicitly) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .with_velocity(Float3(0.0f, 0.0f, 1.0f))
                    .with_acceleration(Float3(0.0f, 0.0f, 2.0f))
                    .build();

    unit.update(1.0f);

    // Symplectic Euler: v becomes 1 + a*dt = 3, then displaces by v*dt = 3.
    EXPECT_NEAR(unit.sync().translation.z, 3.0f, tol);
    expect_vec_near(*unit.velocity(), Float3(0.0f, 0.0f, 3.0f));
}

TEST(Unit, UpdateLeavesStaticUnitUntouched) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(1.0f, 2.0f, 3.0f)))
                    .build();

    unit.update(0.5f);

    expect_vec_near(unit.sync().translation, Float3(1.0f, 2.0f, 3.0f));
}

TEST(Unit, UpdateIgnoresNonPositiveDt) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .with_velocity(Float3(0.0f, 0.0f, 1.0f))
                    .build();

    unit.update(0.0f);
    unit.update(-1.0f);

    expect_vec_near(unit.sync().translation, Float3(0.0f, 0.0f, 0.0f));
}

TEST(Unit, MoveTranslatesPose) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(1.0f, 1.0f, 1.0f)))
                    .build();

    unit.move(Float3(2.0f, 0.0f, -1.0f));

    expect_vec_near(unit.sync().translation, Float3(3.0f, 1.0f, 0.0f));
}

TEST(Unit, SurfaceVelocityIsZeroForStaticUnit) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .build();

    expect_vec_near(unit.surface_velocity(Float3(1.0f, 2.0f, 3.0f)), Float3(0.0f, 0.0f, 0.0f));
}

TEST(Unit, SurfaceVelocityReturnsLinearVelocity) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .with_velocity(Float3(0.0f, 0.0f, 1.0f))
                          .build();

    // A purely linear body has the same surface velocity everywhere.
    expect_vec_near(unit.surface_velocity(Float3(5.0f, -2.0f, 0.0f)), Float3(0.0f, 0.0f, 1.0f));
}

TEST(Unit, SurfaceVelocityAddsRotationalContribution) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                          .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
                          .build();

    // omega x r with omega = (0,0,1) and r = (1,0,0) gives (0,1,0).
    expect_vec_near(unit.surface_velocity(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f));
}

TEST(Unit, WorldBoundReflectsTranslation) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_unit_box())
                          .with_sync(make_sync(Float3(5.0f, 0.0f, 0.0f)))
                          .build();

    const AABB bound = unit.world_bound();

    ASSERT_TRUE(bound.is_valid());
    expect_vec_near(bound.lower_corner, Float3(4.0f, -1.0f, -1.0f));
    expect_vec_near(bound.upper_corner, Float3(6.0f, 1.0f, 1.0f));
}

TEST(Unit, WorldBoundReflectsRotation) {
    // A +90 deg turn about z swaps the box's x and y extents in world space.
    const Unit unit = Unit::builder()
                          .with_geometry(make_box(Float3(-1.0f, -2.0f, -3.0f), Float3(1.0f, 2.0f, 3.0f)))
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f), Quaternion(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f)))
                          .build();

    const AABB bound = unit.world_bound();

    ASSERT_TRUE(bound.is_valid());
    expect_vec_near(bound.lower_corner, Float3(-2.0f, -1.0f, -3.0f), 1e-5f);
    expect_vec_near(bound.upper_corner, Float3(2.0f, 1.0f, 3.0f), 1e-5f);
}

TEST(Unit, TraceReturnsHitInWorldSpace) {
    // The plane sits at local z = 0; the pose lifts it to world z = 5.
    const Unit unit = Unit::builder()
                          .with_geometry(make_plane())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 5.0f)))
                          .build();

    Ray ray;
    ray.origin    = Float3(0.0f, 0.0f, 10.0f);
    ray.direction = Float3(0.0f, 0.0f, -1.0f);

    const HitSurface hit = unit.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 5.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Unit, TraceMissesWhenRayPointsAway) {
    const Unit unit = Unit::builder()
                          .with_geometry(make_plane())
                          .with_sync(make_sync(Float3(0.0f, 0.0f, 5.0f)))
                          .build();

    Ray ray;
    ray.origin    = Float3(0.0f, 0.0f, 10.0f);
    ray.direction = Float3(0.0f, 0.0f, 1.0f);

    EXPECT_FALSE(unit.trace(ray).is_intersecting);
}
