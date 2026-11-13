#include <atlas/unit/unit.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

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

Unit
advance_oscillation_on_backend(const Unit& unit, const float dt) {
    atlas::DeviceBuffer<Unit> units(1, unit);
    auto* data = atlas::raw_pointer_cast(units.data());
    atlas::parallel_for<atlas::ExecutionPolicy::device>(0, 1, [=] ATLAS_ALL_DEVICE (int index) {
        data[index].update(dt);
    });
    const atlas::HostBuffer<Unit> result(units.begin(), units.end());
    return result.front();
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

TEST(Unit, RotateAboutZTurnsPoseDirection) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .build();

    // A +90 deg rotation about world z maps a local +x direction to world +y.
    unit.rotate(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f);

    expect_vec_near(unit.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
}

TEST(Unit, RotateIgnoresAZeroLengthAxis) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .build();

    // A degenerate axis is a guarded no-op: the identity orientation is preserved.
    unit.rotate(Float3(0.0f, 0.0f, 0.0f), pi * 0.5f);

    expect_vec_near(unit.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f), 1e-5f);
}

TEST(Unit, UpdateRotatesPoseByAngularVelocity) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
                    .build();

    // omega magnitude 1 rad/s over dt = pi/2 is a quarter turn about z.
    unit.update(pi * 0.5f);

    expect_vec_near(unit.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
}

TEST(Unit, SetSyncReplacesThePose) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .build();

    unit.set_sync(Sync::builder().with_rigid_pose(Float3(9.0f, 8.0f, 7.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f)).build());

    expect_vec_near(unit.sync().translation, Float3(9.0f, 8.0f, 7.0f));
}

TEST(Unit, SetGeometryChangesTheWorldBound) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
                    .build();

    unit.set_geometry(make_box(Float3(-2.0f, -2.0f, -2.0f), Float3(2.0f, 2.0f, 2.0f)));

    const AABB bound = unit.world_bound();
    ASSERT_TRUE(bound.is_valid());
    expect_vec_near(bound.lower_corner, Float3(-2.0f, -2.0f, -2.0f));
    expect_vec_near(bound.upper_corner, Float3(2.0f, 2.0f, 2.0f));
}

TEST(Unit, WorldBoundOfDefaultGeometryEnclosesTheUnitSphere) {
    // Geometry's default leaf is a unit Sphere, so a default-constructed Unit has a
    // valid local bound and world_bound() reports a valid box that contains the
    // origin (the sphere's center under the identity sync).
    const Unit unit {};

    const AABB bound = unit.world_bound();
    ASSERT_TRUE(bound.is_valid());
    EXPECT_TRUE(bound.contains(Float3(0.0f, 0.0f, 0.0f)));
}


TEST(Unit, AngularOscillationClosesAndReturnsToInitialPose) {
    const auto initial = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), 0.3f);
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(2.0f, 3.0f, 4.0f), initial))
                    .with_angular_oscillation(Float3(0.0f, 0.0f, 3.0f), pi * 0.5f, 2.0f)
                    .build();
    EXPECT_TRUE(unit.dynamic());
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f));
    const Float3 direction(1.0f, 0.2f, 0.3f);
    const auto initial_direction = initial.rotate(direction);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), initial_direction);

    unit.update(1.0f);
    const auto closed = Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), closed.rotate(initial_direction), 1e-5f);
    expect_vec_near(unit.sync().translation, Float3(2.0f, 3.0f, 4.0f));
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f), 1e-5f);

    unit.update(1.0f);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), initial_direction, 1e-5f);
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f));
}

TEST(Unit, AngularOscillationSurfaceVelocityUsesSignedDerivative) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(2.0f, 0.0f, 0.0f)))
                    .with_velocity(Float3(1.0f, 0.0f, 0.0f))
                    .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), -0.4f, 2.0f)
                    .build();
    unit.update(0.5f);
    const float rate = -0.2f * pi;
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f, 0.0f, rate));
    expect_vec_near(unit.surface_velocity(unit.sync().translation + Float3(2.0f, 0.0f, 0.0f)),
                    Float3(1.0f, 2.0f * rate, 0.0f));
    expect_vec_near(*unit.angular_acceleration(), Float3(0.0f), 1e-5f);
    expect_vec_near(unit.sync().translation, Float3(2.5f, 0.0f, 0.0f));

    unit.update(1.0f);
    expect_vec_near(*unit.angular_velocity(), Float3(0.0f, 0.0f, -rate));
    expect_vec_near(unit.surface_velocity(unit.sync().translation + Float3(2.0f, 0.0f, 0.0f)),
                    Float3(1.0f, -2.0f * rate, 0.0f));
}

TEST(Unit, AngularOscillationReturnsAcrossRepeatedCyclesAndLargeSteps) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f)))
                    .with_angular_oscillation(Float3(0.0f, 1.0f, 0.0f), 0.7f, 1.0f)
                    .build();
    for (int step = 0; step < 8000; ++step) unit.update(0.125f);
    expect_vec_near(unit.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    unit.update(1024.25f);
    const auto expected = Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), 0.35f);
    expect_vec_near(unit.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)),
                    expected.rotate(Float3(1.0f, 0.0f, 0.0f)), 1e-5f);
}

TEST(Unit, AngularOscillationRunsOnConfiguredBackend) {
    static_assert(std::is_trivially_copyable_v<Unit>);
    const auto initial = Unit::builder()
                             .with_geometry(make_unit_box())
                             .with_sync(make_sync(Float3(1.0f, 2.0f, 3.0f)))
                             .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f, 1.0f)
                             .build();
    const auto closed = advance_oscillation_on_backend(initial, 0.5f);
    expect_vec_near(closed.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)),
                    Float3(0.0f, 1.0f, 0.0f), 1e-5f);
    const auto open = advance_oscillation_on_backend(closed, 0.5f);
    expect_vec_near(open.sync().sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f), 1e-5f);
}

TEST(Unit, AngularOscillationRejectsInvalidParameters) {
    const auto build = [](const Float3& axis, const float amplitude, const float period) {
        return Unit::builder()
            .with_geometry(make_unit_box())
            .with_sync(make_sync(Float3(0.0f)))
            .with_angular_oscillation(axis, amplitude, period)
            .build();
    };
    const auto axis = Float3(0.0f, 0.0f, 1.0f);
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(build(Float3(0.0f), 1.0f, 1.0f)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(build(Float3(inf, 0.0f, 0.0f), 1.0f, 1.0f)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(build(Float3(0.0f, nan, 0.0f), 1.0f, 1.0f)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(build(axis, inf, 1.0f)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(build(axis, nan, 1.0f)), std::runtime_error);
    for (const float period : { 0.0f, -1.0f, inf, nan, std::numeric_limits<float>::min() }) {
        EXPECT_THROW(static_cast<void>(build(axis, 1.0f, period)), std::runtime_error);
    }
    EXPECT_NO_THROW(static_cast<void>(build(Float3(std::numeric_limits<float>::max(), 0.0f, 0.0f), 0.0f, 1.0f)));
}

TEST(Unit, AngularOscillationRejectsConflictingAngularKinematics) {
    auto builder = Unit::builder()
                       .with_geometry(make_unit_box())
                       .with_sync(make_sync(Float3(0.0f)))
                       .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f);
    EXPECT_THROW(static_cast<void>(builder.with_angular_velocity(Float3(0.0f)).build()), std::runtime_error);
    auto acceleration_builder = Unit::builder()
                                    .with_geometry(make_unit_box())
                                    .with_sync(make_sync(Float3(0.0f)))
                                    .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f);
    EXPECT_THROW(static_cast<void>(acceleration_builder.with_angular_acceleration(Float3(0.0f)).build()), std::runtime_error);
}

TEST(Unit, AngularOscillationPoseEditsPreservePhaseWithoutSnapBack) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f)))
                    .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f)
                    .build();
    unit.update(0.25f);
    const auto new_pose = Sync::builder()
                             .with_rigid_pose(Float3(3.0f), Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), 0.8f))
                             .build();
    unit.set_sync(new_pose);
    unit.update(1.0f);
    const Float3 direction(1.0f, 0.0f, 0.0f);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), new_pose.sync_dir_to_world(direction), 1e-5f);
    unit.rotate(Float3(1.0f, 0.0f, 0.0f), 0.5f);
    const auto rotated = unit.sync();
    unit.update(1.0f);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), rotated.sync_dir_to_world(direction), 1e-5f);
}

TEST(Unit, AngularOscillationNormalizesVeryLargeAndSmallAxes) {
    const Float3 direction(1.0f, 0.0f, 0.0f);
    for (const float scale : { std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min() }) {
        Unit unit = Unit::builder()
                        .with_geometry(make_unit_box())
                        .with_sync(make_sync(Float3(0.0f)))
                        .with_angular_oscillation(Float3(0.0f, scale, scale), pi * 0.5f, 2.0f)
                        .build();
        unit.update(1.0f);
        const Float3 expected(0.0f, 1.0f / atlas::SQRT_TWO, -1.0f / atlas::SQRT_TWO);
        expect_vec_near(unit.sync().sync_dir_to_world(direction), expected, 1e-5f);
    }
}

TEST(Unit, AngularOscillationContinuesFromAnEditedPoseAtTheNextPhase) {
    Unit unit = Unit::builder()
                    .with_geometry(make_unit_box())
                    .with_sync(make_sync(Float3(0.0f)))
                    .with_angular_oscillation(Float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f)
                    .build();
    unit.update(0.25f);
    unit.rotate(Float3(1.0f, 0.0f, 0.0f), 0.7f);
    const Float3 direction(1.0f, 0.2f, 0.3f);
    const Float3 edited = unit.sync().sync_dir_to_world(direction);
    unit.update(0.25f);
    const auto increment = Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), 0.5f);
    expect_vec_near(unit.sync().sync_dir_to_world(direction), increment.rotate(edited), 1e-5f);
}
