#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Sync, DefaultConstructorBuildsIdentityPoseAndActsAsNoOp) {

    // Construct Sync<T> using the default constructor.
    // Contract:
    // - Sync<T> must be immediately usable after default construction.
    // - The default state represents the identity pose:
    //     translation = (0,0,0)
    //     orientation = identity quaternion
    // - Therefore, all sync operations must behave as no-ops.
    const system::Sync<double> s;

    // Use a non-trivial point (mixed signs + fractional component) so that
    // accidental translation/rotation cannot "pass" silently.
    const Vector3<double> point(1.0, -2.0, 3.5);

    // Use a non-zero direction with varied components to exercise direction paths.
    // Directions must never be affected by translation; for the identity pose,
    // they also must not be rotated.
    const Vector3<double> dir(0.5, 2.0, -4.0);

    // -------------------------
    // Point sync (return-by-value)
    // -------------------------

    EXPECT_TRUE(test::vec_near(s.sync_to_world(point), point, eps));
    EXPECT_TRUE(test::vec_near(s.sync_to_local(point), point, eps));

    // -------------------------
    // Direction sync (return-by-value)
    // -------------------------

    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(dir), dir, eps));
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_local(dir), dir, eps));

    // -------------------------
    // Ray sync (origin + direction)
    // -------------------------

    const Ray<double> ray(
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(0.0, 0.0, 1.0));

    const auto world_ray = s.sync_to_world(ray);
    const auto local_ray = s.sync_to_local(ray);

    EXPECT_TRUE(test::vec_near(world_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(world_ray.direction, ray.direction, eps));

    EXPECT_TRUE(test::vec_near(local_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(local_ray.direction, ray.direction, eps));
}

TEST(Sync, PoseConstructorBuildsPoseAndAppliesTranslationAndRotation) {

    // -------------------------
    // Arrange: provide a full pose (translation + non-identity rotation)
    // -------------------------

    // Translation component of the pose.
    // This must affect points (after rotation), but must never affect directions.
    const Vector3<double> t(-4.0, 5.0, 6.0);

    // Build a non-identity rotation so we can observe rotation effects clearly.
    // We choose +90 degrees about +Z because the expected mapping is simple:
    //   (1,0,0) -> (0,1,0)
    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    // Construct Sync<T> from (translation, orientation).
    // Contract:
    // - Sync<T> stores a rigid pose.
    // - Points use:     world = R(q) * local + t
    // - Directions use: world = R(q) * local
    const system::Sync<double> s(t, q);

    // -------------------------
    // Act + Assert: direction transforms (rotation only)
    // -------------------------

    const Vector3<double> local_dir(1.0, 0.0, 0.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    const auto world_dir = s.sync_dir_to_world(local_dir);
    EXPECT_TRUE(test::vec_near(world_dir, expected_dir, eps));

    const auto back_dir = s.sync_dir_to_local(world_dir);
    EXPECT_TRUE(test::vec_near(back_dir, local_dir, eps));

    // -------------------------
    // Act + Assert: point transforms (rotation + translation)
    // -------------------------

    const Vector3<double> local_point(1.0, 0.0, 0.0);
    const Vector3<double> expected_point(-4.0, 6.0, 6.0);

    const auto world_point = s.sync_to_world(local_point);
    EXPECT_TRUE(test::vec_near(world_point, expected_point, eps));

    const auto back_point = s.sync_to_local(world_point);
    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));
}

TEST(Sync, RayTransformMatchesRigidRulesOriginRotatesAndTranslatesDirectionRotatesOnly) {

    // -------------------------
    // Arrange: build a rigid pose (translation + non-identity rotation)
    // -------------------------

    const Vector3<double> t(10.0, -2.0, 3.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    // Expected rigid rules:
    // - Points:     world = R(q) * local + t
    // - Directions: world = R(q) * local
    const system::Sync<double> s(t, q);

    // -------------------------
    // Arrange: local ray input
    // -------------------------

    const Ray<double> local(
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0));

    // -------------------------
    // Expected world ray
    // -------------------------

    const Vector3<double> expected_origin(10.0, -1.0, 3.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    // -------------------------
    // Act: local → world
    // -------------------------

    const auto world = s.sync_to_world(local);

    // -------------------------
    // Assert
    // -------------------------

    EXPECT_TRUE(test::vec_near(world.origin, expected_origin, eps));
    EXPECT_TRUE(test::vec_near(world.direction, expected_dir, eps));

    // -------------------------
    // Act: world → local
    // -------------------------

    const auto back = s.sync_to_local(world);

    EXPECT_TRUE(test::vec_near(back.origin, local.origin, eps));
    EXPECT_TRUE(test::vec_near(back.direction, local.direction, eps));
}

TEST(Sync, SyncOperatorExposureIsConsistentWithWrapperResults) {

    // -------------------------
    // Arrange: create a Sync<T> with a non-trivial pose
    // -------------------------

    const Vector3<double> t(1.0, 2.0, 3.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    // The test verifies that Sync<T> delegates to the underlying SyncOperator<T>
    // without adding inconsistent behavior.
    const system::Sync<double> s(t, q);

    // -------------------------
    // Arrange: inputs
    // -------------------------

    const Vector3<double> p_local(1.0, 0.0, 0.0);
    const Vector3<double> d_local(1.0, 0.0, 0.0);

    // -------------------------
    // Act: wrapper API
    // -------------------------

    const auto p_world_a = s.sync_to_world(p_local);
    const auto d_world_a = s.sync_dir_to_world(d_local);

    // -------------------------
    // Act: underlying operator API
    // -------------------------

    Vector3<double> p_world_b;
    Vector3<double> d_world_b;

    s.sync().sync_to_world(p_local, p_world_b);
    s.sync().sync_dir_to_world(d_local, d_world_b);

    // -------------------------
    // Assert
    // -------------------------

    EXPECT_TRUE(test::vec_near(p_world_a, p_world_b, eps));
    EXPECT_TRUE(test::vec_near(d_world_a, d_world_b, eps));
}

TEST(Sync, RebuildMatricesIsIdempotentAndDoesNotChangeResults) {

    // -------------------------
    // Arrange: create a Sync<T> with a unit rotation and non-trivial translation
    // -------------------------

    const Vector3<double> t(3.0, -4.0, 5.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    // Non-const because we will call rebuild_matrices().
    system::Sync<double> s(t, q);

    // -------------------------
    // Arrange: inputs
    // -------------------------

    const Vector3<double> p_local(1.0, 2.0, 3.0);
    const Vector3<double> d_local(1.0, -1.0, 0.5);
    const Ray<double> r_local(
        Vector3<double>(-2.0, 0.25, 1.5),
        Vector3<double>(0.0, 1.0, 0.0));

    // -------------------------
    // Act: compute baseline outputs BEFORE rebuild
    // -------------------------

    const auto p_world_0 = s.sync_to_world(p_local);
    const auto p_back_0  = s.sync_to_local(p_world_0);

    const auto d_world_0 = s.sync_dir_to_world(d_local);
    const auto d_back_0  = s.sync_dir_to_local(d_world_0);

    const auto r_world_0 = s.sync_to_world(r_local);
    const auto r_back_0  = s.sync_to_local(r_world_0);

    // -------------------------
    // Assert: baseline sanity
    // -------------------------

    EXPECT_TRUE(test::vec_near(p_back_0, p_local, eps));
    EXPECT_TRUE(test::vec_near(d_back_0, d_local, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.origin, r_local.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.direction, r_local.direction, eps));

    // -------------------------
    // Act: rebuild cached matrices
    // -------------------------

    s.rebuild_matrices();

    // -------------------------
    // Act: compute outputs AFTER rebuild
    // -------------------------

    const auto p_world_1 = s.sync_to_world(p_local);
    const auto p_back_1  = s.sync_to_local(p_world_1);

    const auto d_world_1 = s.sync_dir_to_world(d_local);
    const auto d_back_1  = s.sync_dir_to_local(d_world_1);

    const auto r_world_1 = s.sync_to_world(r_local);
    const auto r_back_1  = s.sync_to_local(r_world_1);

    // -------------------------
    // Assert: idempotency / consistency
    // -------------------------

    EXPECT_TRUE(test::vec_near(p_world_1, p_world_0, eps));
    EXPECT_TRUE(test::vec_near(p_back_1, p_back_0, eps));

    EXPECT_TRUE(test::vec_near(d_world_1, d_world_0, eps));
    EXPECT_TRUE(test::vec_near(d_back_1, d_back_0, eps));

    EXPECT_TRUE(test::vec_near(r_world_1.origin, r_world_0.origin, eps));
    EXPECT_TRUE(test::vec_near(r_world_1.direction, r_world_0.direction, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.origin, r_back_0.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.direction, r_back_0.direction, eps));

    // -------------------------
    // Assert: round-trip still holds
    // -------------------------

    EXPECT_TRUE(test::vec_near(p_back_1, p_local, eps));
    EXPECT_TRUE(test::vec_near(d_back_1, d_local, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.origin, r_local.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.direction, r_local.direction, eps));
}

TEST(Sync, SetOrientationRebuildsAndUpdatesRotation) {

    // -------------------------
    // Arrange: start from identity rotation
    // -------------------------

    // Use a non-zero translation to ensure we are in a realistic pose setup.
    // Translation must not affect direction vectors.
    const Vector3<double> t(1.0, 2.0, 3.0);

    // Default quaternion is assumed to represent identity rotation.
    const math::Quaternion<double> q_identity;

    system::Sync<double> s(t, q_identity);

    const Vector3<double> x(1.0, 0.0, 0.0);

    // With identity rotation, local +X must remain +X in world space.
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(x), x, eps));

    // -------------------------
    // Act: update orientation
    // -------------------------

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q_rot(axis, radians);

    // set_orientation() must update the stored orientation and rebuild
    // cached matrices immediately.
    s.set_orientation(q_rot);

    // -------------------------
    // Assert
    // -------------------------

    const Vector3<double> expected_y(0.0, 1.0, 0.0);
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(x), expected_y, eps));
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_local(expected_y), x, eps));
}

TEST(Sync, SetPoseRebuildsAndUpdatesBothTranslationAndRotation) {

    // -------------------------
    // Arrange: start from the identity pose
    // -------------------------

    const Vector3<double> t0(0.0, 0.0, 0.0);
    const math::Quaternion<double> q0;

    system::Sync<double> s(t0, q0);

    // New pose to apply: translation + non-identity rotation.
    const Vector3<double> t_new(-4.0, 5.0, 6.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q_new(axis, radians);

    // -------------------------
    // Act: update the full pose
    // -------------------------

    // set_pose() must assign translation and orientation, then rebuild caches.
    s.set_pose(t_new, q_new);

    const Vector3<double> local_point(1.0, 0.0, 0.0);

    // -------------------------
    // Assert
    // -------------------------

    const Vector3<double> expected_ccw(-4.0, 6.0, 6.0);
    const Vector3<double> expected_cw(-4.0, 4.0, 6.0);

    const auto got = s.sync_to_world(local_point);

    EXPECT_TRUE(
        test::vec_near(got, expected_ccw, eps) || test::vec_near(got, expected_cw, eps));

    EXPECT_TRUE(test::vec_near(s.sync_to_local(got), local_point, eps));
}