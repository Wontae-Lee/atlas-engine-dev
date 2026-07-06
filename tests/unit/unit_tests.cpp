#include <atlas/unit/unit.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/sync/sync.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::Quaternion;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Vector3;
using atlas::pi;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

Geometry
make_geometry() {
    return Geometry(Box::builder()
                                .with_lower_corner(Vector3(-1.0f, -1.0f, -1.0f))
                                .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
                                .build());
}

SyncHostPtr
make_sync() {
    return Sync::builder()
        .with_rigid_pose(Vector3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

}

TEST(Unit, BuilderConstructsStaticUnit) {
    const auto unit = Unit::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .build();

    EXPECT_FALSE(unit.dynamic());
}

TEST(Unit, BuilderConstructsDynamicUnitAndCanonicalizesKinematics) {
    const auto unit = Unit::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .with_velocity(Vector3(1.0f, 0.0f, 0.0f))
                          .build();

    EXPECT_TRUE(unit.dynamic());
    ASSERT_TRUE(unit.velocity().has_value());
    ASSERT_TRUE(unit.acceleration().has_value());
    expect_vec_near(*unit.acceleration(), Vector3(0.0f, 0.0f, 0.0f));
}

TEST(Unit, BuilderRejectsMissingDependencies) {
    EXPECT_THROW(static_cast<void>(Unit::builder()
                                       .with_sync(make_sync())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(Unit::builder()
                                       .with_geometry(make_geometry())
                                       .build()),
                 std::runtime_error);
}

TEST(Unit, UpdateAppliesVelocity) {
    auto unit = Unit::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .with_velocity(Vector3(1.0f, 0.0f, 0.0f))
                    .build();

    unit.update(1.0f);

    expect_vec_near(unit.sync().translation, Vector3(1.0f, 0.0f, 0.0f));
}

TEST(Unit, GeometryOperatorRemainsValidAfterGeometryOwnerDies) {
    Unit unit;

    {
        const auto geometry = make_geometry();
        unit                = Unit::builder()
                   .with_geometry(geometry)
                   .with_sync(make_sync())
                   .build();
    }

    const auto& op = unit.geometry();

    EXPECT_TRUE(op.is_valid());
    EXPECT_TRUE(op.is_inside(Vector3(0.0f, 0.0f, 0.0f)));
    expect_vec_near(op.centroid(), Vector3(0.0f, 0.0f, 0.0f));
}

TEST(Unit, MoveAndRotateUpdatePose) {
    auto unit = Unit::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .build();

    unit.move(Vector3(1.0f, 2.0f, 3.0f));
    unit.rotate(Vector3(0.0f, 0.0f, 1.0f), pi / 2.0f);

    expect_vec_near(unit.sync().translation, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(unit.sync().sync_dir_to_world(Vector3(1.0f, 0.0f, 0.0f)), Vector3(0.0f, 1.0f, 0.0f));
}
