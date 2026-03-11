#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>

using namespace atlas;

TEST(Unit, RotateWithZeroAxisIsNoOp) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(1.0, 2.0, 3.0);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    const auto before = u.sync_operator();

    u.rotate(Vector3<double>(0.0, 0.0, 0.0), M_PI / 2.0);

    const auto after = u.sync_operator();

    EXPECT_TRUE(test::vec_near(after.translation, before.translation, eps));
    EXPECT_NEAR(after.orientation.w, before.orientation.w, eps);
    EXPECT_NEAR(after.orientation.x, before.orientation.x, eps);
    EXPECT_NEAR(after.orientation.y, before.orientation.y, eps);
    EXPECT_NEAR(after.orientation.z, before.orientation.z, eps);
}

TEST(Unit, RotateWithZeroAngleKeepsOrientationUnchanged) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(-1.0, 4.0, 2.0);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    const auto before = u.sync_operator();

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const auto after = u.sync_operator();

    EXPECT_TRUE(test::vec_near(after.translation, before.translation, eps));
    EXPECT_NEAR(after.orientation.w, before.orientation.w, eps);
    EXPECT_NEAR(after.orientation.x, before.orientation.x, eps);
    EXPECT_NEAR(after.orientation.y, before.orientation.y, eps);
    EXPECT_NEAR(after.orientation.z, before.orientation.z, eps);
}

TEST(Unit, RotateFromIdentityPoseUpdatesDirectionAsExpected) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(1.0, 0.0, 0.0);

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);

    const auto world_dir = u.sync_operator().sync_dir_to_world(local_dir);

    EXPECT_TRUE(test::vec_near(world_dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, RotateDoesNotModifyTranslation) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(3.0, -2.0, 7.0);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t0, eps));
}

TEST(Unit, RotateNormalizesAxisBeforeApplyingRotation) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u1(qop, top, system::SyncOperator<double> {});
    system::Unit<double> u2(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(1.0, 0.0, 0.0);

    u1.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);
    u2.rotate(Vector3<double>(0.0, 0.0, 5.0), M_PI / 2.0);

    const auto world_dir_1 = u1.sync_operator().sync_dir_to_world(local_dir);
    const auto world_dir_2 = u2.sync_operator().sync_dir_to_world(local_dir);

    EXPECT_TRUE(test::vec_near(world_dir_1, world_dir_2, eps));
    EXPECT_TRUE(test::vec_near(world_dir_1, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, RotateChangesWorldPointButPreservesDistanceFromTranslationOrigin) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(2.0, -1.0, 0.5);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_point(1.0, 0.0, 0.0);

    const auto before_world    = u.sync_operator().sync_to_world(local_point);
    const double before_offset = (before_world - u.sync_operator().translation).length();

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);

    const auto after_world    = u.sync_operator().sync_to_world(local_point);
    const double after_offset = (after_world - u.sync_operator().translation).length();

    EXPECT_TRUE(test::vec_near(after_world, Vector3<double>(2.0, 0.0, 0.5), eps));
    EXPECT_NEAR(after_offset, before_offset, eps);
}

TEST(Unit, RotateAndInverseSyncRemainConsistentForPoints) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(2.0, -1.0, 0.5);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_point(2.0, -3.0, 4.0);

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 3.0);

    const auto world_point = u.sync_operator().sync_to_world(local_point);
    const auto back_point  = u.sync_operator().sync_to_local(world_point);

    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));
}

TEST(Unit, RotateAndInverseSyncRemainConsistentForDirections) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(1.0, 2.0, -3.0);

    u.rotate(Vector3<double>(1.0, 1.0, 1.0), M_PI / 4.0);

    const auto world_dir = u.sync_operator().sync_dir_to_world(local_dir);
    const auto back_dir  = u.sync_operator().sync_dir_to_local(world_dir);

    EXPECT_TRUE(test::vec_near(back_dir, local_dir, eps));
}

TEST(Unit, ConsecutiveRotationsAccumulateInOrder) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(1.0, 0.0, 0.0);

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);
    const auto after_first = u.sync_operator().sync_dir_to_world(local_dir);
    EXPECT_TRUE(test::vec_near(after_first, Vector3<double>(0.0, 1.0, 0.0), eps));

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);
    const auto after_second = u.sync_operator().sync_dir_to_world(local_dir);
    EXPECT_TRUE(test::vec_near(after_second, Vector3<double>(-1.0, 0.0, 0.0), eps));
}

TEST(Unit, RotateByAngleAndThenNegativeAngleRestoresOriginalOrientationEffect) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(0.25, 1.5, -2.0);
    const auto before = u.sync_operator().sync_dir_to_world(local_dir);

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 3.0);
    u.rotate(Vector3<double>(0.0, 0.0, 1.0), -M_PI / 3.0);

    const auto after = u.sync_operator().sync_dir_to_world(local_dir);

    EXPECT_TRUE(test::vec_near(after, before, eps));
}

TEST(Unit, RotateAffectsRayOriginAndDirectionThroughSyncOperator) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Ray<double> local_ray(
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0));

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);

    const auto world_ray = u.sync_operator().sync_to_world(local_ray);

    EXPECT_TRUE(test::vec_near(world_ray.origin, Vector3<double>(0.0, 1.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(world_ray.direction, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, RotateKeepsDirectionLengthInvariant) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    system::Unit<double> u(qop, top, system::SyncOperator<double> {});

    const Vector3<double> local_dir(1.0, -2.0, 3.0);
    const double len_before = local_dir.length();

    u.rotate(Vector3<double>(0.0, 1.0, 0.0), M_PI / 4.0);

    const auto world_dir   = u.sync_operator().sync_dir_to_world(local_dir);
    const double len_after = world_dir.length();

    EXPECT_NEAR(len_after, len_before, eps);
}

TEST(Unit, RotateKeepsPointOffsetLengthInvariantWhenTranslationIsFixed) {
    const auto sphere = test::make_sphere();
    const auto qop    = sphere.make_query_operator();
    const auto top    = sphere.make_trace_operator();

    const Vector3<double> t0(4.0, -1.0, 2.0);
    const math::Quaternion<double> q0;
    system::Unit<double> u(qop, top, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_point(1.0, 2.0, -2.0);

    const auto before_world    = u.sync_operator().sync_to_world(local_point);
    const double offset_before = (before_world - u.sync_operator().translation).length();

    u.rotate(Vector3<double>(0.0, 0.0, 1.0), M_PI / 5.0);

    const auto after_world    = u.sync_operator().sync_to_world(local_point);
    const double offset_after = (after_world - u.sync_operator().translation).length();

    EXPECT_NEAR(offset_after, offset_before, eps);
}