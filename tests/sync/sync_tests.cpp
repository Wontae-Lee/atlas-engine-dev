#include <atlas/sync/sync.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Quaternion;
using atlas::Ray;
using atlas::Sync;
using atlas::Vector3;
using atlas::pi;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Sync, DefaultConstructorCreatesIdentityTransform) {
    const Sync sync;

    expect_vec_near(sync.translation, Vector3(0.0f, 0.0f, 0.0f));
    expect_vec_near(sync.sync_to_world(Vector3(1.0f, 2.0f, 3.0f)), Vector3(1.0f, 2.0f, 3.0f));
}

TEST(Sync, BuilderConstructsConfiguredSync) {
    const auto sync = Sync::builder()
                          .with_rigid_pose(Vector3(1.0f, 2.0f, 3.0f),
                                           Quaternion::from_axis_angle(Vector3(0.0f, 0.0f, 1.0f), pi / 2.0f))
                          .build();

    expect_vec_near(sync.sync_to_world(Vector3(1.0f, 0.0f, 0.0f)), Vector3(1.0f, 3.0f, 3.0f));
}

TEST(Sync, BuilderRejectsInvalidQuaternion) {
    EXPECT_THROW(static_cast<void>(Sync::builder()
                                       .with_rigid_pose(Vector3(0.0f, 0.0f, 0.0f), Quaternion(0.0f, 0.0f, 0.0f, 0.0f))
                                       .build()),
                 std::runtime_error);
}

TEST(Sync, SettersUpdatePose) {
    Sync sync;

    sync.set_translation(Vector3(1.0f, 2.0f, 3.0f));
    sync.set_orientation(Quaternion::from_axis_angle(Vector3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    expect_vec_near(sync.sync_to_world(Vector3(1.0f, 0.0f, 0.0f)), Vector3(1.0f, 3.0f, 3.0f));
}

TEST(Sync, MakeHostSharedBuildsSync) {
    const auto sync = Sync::builder()
                          .with_rigid_pose(Vector3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                          .make_host_shared();

    ASSERT_NE(sync, nullptr);
    expect_vec_near(sync->sync_to_world(Vector3(1.0f, 0.0f, 0.0f)), Vector3(1.0f, 0.0f, 0.0f));
}

TEST(Sync, PointTransformsRoundTrip) {
    const Sync sync(
        Vector3(1.0f, 2.0f, 3.0f),
        Quaternion::from_axis_angle(Vector3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    const Vector3 local(1.0f, 0.0f, 0.0f);
    const Vector3 world     = sync.sync_to_world(local);
    const Vector3 roundtrip = sync.sync_to_local(world);

    expect_vec_near(roundtrip, local);
}

TEST(Sync, DirectionTransformsIgnoreTranslation) {
    const Sync sync(
        Vector3(5.0f, 6.0f, 7.0f),
        Quaternion::from_axis_angle(Vector3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    const Vector3 world_dir = sync.sync_dir_to_world(Vector3(1.0f, 0.0f, 0.0f));

    EXPECT_NEAR(world_dir.length(), 1.0f, tol);
    expect_vec_near(world_dir, Vector3(0.0f, 1.0f, 0.0f));
}

TEST(Sync, RayTransformsRoundTrip) {
    const Sync sync(
        Vector3(1.0f, 0.0f, 0.0f),
        Quaternion::from_axis_angle(Vector3(0.0f, 1.0f, 0.0f), pi / 2.0f));
    const Ray local(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f));

    const Ray world    = sync.sync_to_world(local);
    const Ray restored = sync.sync_to_local(world);

    expect_vec_near(restored.origin, local.origin);
    expect_vec_near(restored.direction, local.direction);
}
