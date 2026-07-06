#include <atlas/sync/sync.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Quaternion;
using atlas::Ray;
using atlas::Sync;
using atlas::Float3;
using atlas::pi;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Sync, DefaultConstructorCreatesIdentityTransform) {
    const Sync sync;

    expect_vec_near(sync.translation, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(sync.sync_to_world(Float3(1.0f, 2.0f, 3.0f)), Float3(1.0f, 2.0f, 3.0f));
}

TEST(Sync, BuilderConstructsConfiguredSync) {
    const auto sync = Sync::builder()
                          .with_rigid_pose(Float3(1.0f, 2.0f, 3.0f),
                                           Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi / 2.0f))
                          .build();

    expect_vec_near(sync.sync_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 3.0f, 3.0f));
}

TEST(Sync, BuilderRejectsInvalidQuaternion) {
    EXPECT_THROW(static_cast<void>(Sync::builder()
                                       .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(0.0f, 0.0f, 0.0f, 0.0f))
                                       .build()),
                 std::runtime_error);
}

TEST(Sync, SettersUpdatePose) {
    Sync sync;

    sync.set_translation(Float3(1.0f, 2.0f, 3.0f));
    sync.set_orientation(Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    expect_vec_near(sync.sync_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 3.0f, 3.0f));
}

TEST(Sync, MakeHostSharedBuildsSync) {
    const auto sync = Sync::builder()
                          .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                          .make_host_shared();

    ASSERT_NE(sync, nullptr);
    expect_vec_near(sync->sync_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
}

TEST(Sync, PointTransformsRoundTrip) {
    const Sync sync(
        Float3(1.0f, 2.0f, 3.0f),
        Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    const Float3 local(1.0f, 0.0f, 0.0f);
    const Float3 world     = sync.sync_to_world(local);
    const Float3 roundtrip = sync.sync_to_local(world);

    expect_vec_near(roundtrip, local);
}

TEST(Sync, DirectionTransformsIgnoreTranslation) {
    const Sync sync(
        Float3(5.0f, 6.0f, 7.0f),
        Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi / 2.0f));

    const Float3 world_dir = sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f));

    EXPECT_NEAR(world_dir.length(), 1.0f, tol);
    expect_vec_near(world_dir, Float3(0.0f, 1.0f, 0.0f));
}

TEST(Sync, RayTransformsRoundTrip) {
    const Sync sync(
        Float3(1.0f, 0.0f, 0.0f),
        Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), pi / 2.0f));
    const Ray local(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f));

    const Ray world    = sync.sync_to_world(local);
    const Ray restored = sync.sync_to_local(world);

    expect_vec_near(restored.origin, local.origin);
    expect_vec_near(restored.direction, local.direction);
}
