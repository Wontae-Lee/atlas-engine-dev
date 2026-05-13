#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryHostPtr;
using atlas::QuaternionF;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Vector3F;
using atlas::pi;
using atlas::test::vec_near;
using atlas::tol;

GeometryHostPtr<float>
make_geometry() {
    return Box<float>::builder()
        .with_lower_corner(Vector3F(-1, -1, -1))
        .with_upper_corner(Vector3F(1, 1, 1))
        .make_host_shared();
}

SyncHostPtr<float>
make_sync() {
    return Sync<float>::builder()
        .with_rigid_pose(Vector3F(0, 0, 0), QuaternionF(1, 0, 0, 0))
        .make_host_shared();
}

} // namespace

TEST(Unit, BuilderConstructsStaticUnit) {
    // Arrange and act: build a unit with geometry and synchronization only.
    const auto unit = Unit<float>::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .build();

    // Assert: a unit with no kinematics is static.
    EXPECT_FALSE(unit.dynamic());
}

TEST(Unit, BuilderConstructsDynamicUnitAndCanonicalizesKinematics) {
    // Arrange and act: build a unit with velocity but no explicit acceleration.
    const auto unit = Unit<float>::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .with_velocity(Vector3F(1, 0, 0))
                          .build();

    // Assert: kinematic input makes the unit dynamic and fills missing acceleration.
    EXPECT_TRUE(unit.dynamic());
    ASSERT_TRUE(unit.velocity().has_value());
    ASSERT_TRUE(unit.acceleration().has_value());
    EXPECT_TRUE(vec_near(*unit.acceleration(), Vector3F(0, 0, 0), tol));
}

TEST(Unit, BuilderRejectsMissingDependencies) {
    // A unit cannot be built without geometry.
    EXPECT_THROW(Unit<float>::builder()
                     .with_sync(make_sync())
                     .build(),
                 std::runtime_error);

    // A unit cannot be built without synchronization.
    EXPECT_THROW(Unit<float>::builder()
                     .with_geometry(make_geometry())
                     .build(),
                 std::runtime_error);
}

TEST(Unit, UpdateAppliesVelocity) {
    // Arrange: build a dynamic unit with one unit of velocity along +X.
    auto unit = Unit<float>::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .with_velocity(Vector3F(1, 0, 0))
                    .build();

    // Act: advance the unit for one second.
    unit.update(1.0f);

    // Assert: translation follows velocity * dt.
    EXPECT_TRUE(vec_near(unit.sync_operator().translation, Vector3F(1, 0, 0), tol));
}

TEST(Unit, MoveAndRotateUpdatePose) {
    // Arrange: build a static unit with identity pose.
    auto unit = Unit<float>::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .build();

    // Act: move the unit and rotate it 90 degrees around +Z.
    unit.move(Vector3F(1, 2, 3));
    unit.rotate(Vector3F(0, 0, 1), static_cast<float>(pi / 2));

    // Assert: translation and orientation reflect the pose edits.
    EXPECT_TRUE(vec_near(unit.sync_operator().translation, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(unit.sync_operator().sync_dir_to_world(Vector3F(1, 0, 0)), Vector3F(0, 1, 0), tol));
}
