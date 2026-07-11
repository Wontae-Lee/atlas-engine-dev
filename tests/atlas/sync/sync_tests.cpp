#include <atlas/sync/sync.h>

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using atlas::Float3;
using atlas::pi;
using atlas::Quaternion;
using atlas::Ray;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::tol;

// A +90 degree rotation about the world z axis, expressed as a unit quaternion.
Quaternion
quarter_turn_z() {
    return Quaternion(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f);
}

void
expect_vec_near(const Float3& actual, const Float3& expected, const float eps = tol) {
    EXPECT_NEAR(actual.x, expected.x, eps);
    EXPECT_NEAR(actual.y, expected.y, eps);
    EXPECT_NEAR(actual.z, expected.z, eps);
}

}

TEST(Sync, BuilderDefaultsToIdentity) {
    const Sync sync = Sync::builder().build();

    expect_vec_near(sync.translation, Float3(0.0f, 0.0f, 0.0f));
    // With no setter called the pose is the identity, so a point is unmoved.
    expect_vec_near(sync.sync_to_world(Float3(3.0f, -2.0f, 1.0f)), Float3(3.0f, -2.0f, 1.0f));
}

TEST(Sync, BuilderRoundTripsPose) {
    const Sync sync = Sync::builder()
                          .with_rigid_pose(Float3(1.0f, 2.0f, 3.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                          .build();

    expect_vec_near(sync.translation, Float3(1.0f, 2.0f, 3.0f));
    EXPECT_NEAR(sync.orientation.w, 1.0f, tol);
    EXPECT_NEAR(sync.orientation.x, 0.0f, tol);
    EXPECT_NEAR(sync.orientation.y, 0.0f, tol);
    EXPECT_NEAR(sync.orientation.z, 0.0f, tol);
}

TEST(Sync, BuilderRejectsZeroQuaternion) {
    EXPECT_THROW(
        static_cast<void>(Sync::builder()
                              .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(0.0f, 0.0f, 0.0f, 0.0f))
                              .build()),
        std::runtime_error);
}

TEST(Sync, BuilderRejectsNonFiniteTranslation) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW(
        static_cast<void>(Sync::builder()
                              .with_rigid_pose(Float3(inf, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                              .build()),
        std::runtime_error);
}

TEST(Sync, MakeHostSharedBuildsSync) {
    const SyncHostPtr sync = Sync::builder()
                                 .with_rigid_pose(Float3(4.0f, 5.0f, 6.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
                                 .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(sync));
    expect_vec_near(sync->translation, Float3(4.0f, 5.0f, 6.0f));
}

TEST(Sync, DefaultConstructedPoseIsIdentity) {
    const Sync sync {};

    expect_vec_near(sync.sync_to_world(Float3(7.0f, -3.0f, 2.0f)), Float3(7.0f, -3.0f, 2.0f));
    expect_vec_near(sync.sync_to_local(Float3(7.0f, -3.0f, 2.0f)), Float3(7.0f, -3.0f, 2.0f));
}

TEST(Sync, SyncToWorldAppliesRotationThenTranslation) {
    const Sync sync = Sync::builder()
                          .with_rigid_pose(Float3(2.0f, 3.0f, 4.0f), quarter_turn_z())
                          .build();

    // A +90 deg turn about z maps (1,0,0) to (0,1,0), then the translation adds.
    expect_vec_near(sync.sync_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(2.0f, 4.0f, 4.0f), 1e-5f);
}

TEST(Sync, SyncToLocalInvertsSyncToWorld) {
    const Sync   sync = Sync::builder()
                          .with_rigid_pose(Float3(2.0f, 3.0f, 4.0f), quarter_turn_z())
                          .build();
    const Float3 point(1.5f, -0.5f, 2.0f);

    expect_vec_near(sync.sync_to_local(sync.sync_to_world(point)), point, 1e-5f);
}

TEST(Sync, DirectionTransformIgnoresTranslation) {
    const Sync sync = Sync::builder()
                          .with_rigid_pose(Float3(10.0f, 20.0f, 30.0f), quarter_turn_z())
                          .build();

    // A direction is translation-invariant: only the rotation applies.
    expect_vec_near(sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
    // And the inverse direction rotation round-trips.
    const Float3 dir(0.3f, -0.7f, 0.2f);
    expect_vec_near(sync.sync_dir_to_local(sync.sync_dir_to_world(dir)), dir, 1e-5f);
}

TEST(Sync, RayRoundTripsThroughLocalFrame) {
    const Sync sync = Sync::builder()
                          .with_rigid_pose(Float3(2.0f, 3.0f, 4.0f), quarter_turn_z())
                          .build();

    Ray world_ray;
    world_ray.origin    = Float3(1.0f, 1.0f, 1.0f);
    world_ray.direction = Float3(0.0f, 0.0f, -1.0f);

    const Ray restored = sync.sync_to_world(sync.sync_to_local(world_ray));

    expect_vec_near(restored.origin, world_ray.origin, 1e-5f);
    expect_vec_near(restored.direction, world_ray.direction, 1e-5f);
}

TEST(Sync, SetTranslationLeavesOrientationUntouched) {
    Sync sync = Sync::builder().with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), quarter_turn_z()).build();

    sync.set_translation(Float3(5.0f, 6.0f, 7.0f));

    expect_vec_near(sync.translation, Float3(5.0f, 6.0f, 7.0f));
    // Orientation is unchanged, so the rotation still maps (1,0,0) to (0,1,0).
    expect_vec_near(sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
}

TEST(Sync, SetOrientationRebuildsMatrices) {
    Sync sync {};

    sync.set_orientation(quarter_turn_z());

    // The cached matrices are rebuilt, so a direction transforms as expected.
    expect_vec_near(sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
    // And the inverse matrix is a true inverse: the round trip returns the input.
    const Float3 dir(1.0f, 0.0f, 0.0f);
    expect_vec_near(sync.sync_dir_to_local(sync.sync_dir_to_world(dir)), dir, 1e-5f);
}

TEST(Sync, SetPoseUpdatesTranslationAndOrientation) {
    Sync sync {};

    sync.set_pose(Float3(1.0f, 0.0f, 0.0f), quarter_turn_z());

    expect_vec_near(sync.sync_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(1.0f, 1.0f, 0.0f), 1e-5f);
}

TEST(Sync, ValueConstructorBuildsMatricesImmediately) {
    // The two-argument value constructor must derive the cached rotation matrices
    // eagerly, so a direction transform works without any further setup.
    const Sync sync(Float3(1.0f, 2.0f, 3.0f), quarter_turn_z());

    expect_vec_near(sync.translation, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
}

TEST(Sync, BuilderRejectsNonFiniteOrientation) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW(
        static_cast<void>(Sync::builder()
                              .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(inf, 0.0f, 0.0f, 0.0f))
                              .build()),
        std::runtime_error);
}

TEST(Sync, RebuildMatricesReflectsADirectOrientationWrite) {
    Sync sync {};

    // A direct write to the public orientation field leaves the cached matrices
    // stale until rebuild_matrices() is called to resynchronize them.
    sync.orientation = quarter_turn_z();
    sync.rebuild_matrices();

    expect_vec_near(sync.sync_dir_to_world(Float3(1.0f, 0.0f, 0.0f)), Float3(0.0f, 1.0f, 0.0f), 1e-5f);
}
