#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>

using namespace atlas;

TEST(Unit, MoveFromIdentityPoseUpdatesTranslationExactly) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    system::SyncOperator<double> sop;

    system::Unit<double> u(geometry_op, sop);

    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(0.0, 0.0, 0.0),
        eps));

    const Vector3<double> delta(1.25, -2.5, 3.75);
    u.move(delta);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, delta, eps));
}

TEST(Unit, MoveAccumulatesTranslationFromNonZeroInitialPose) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> t0(1.0, 2.0, 3.0);
    const math::Quaternion<double> q0;

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> delta(-4.0, 5.0, 6.0);
    u.move(delta);

    const Vector3<double> expected = t0 + delta;

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, expected, eps));
}

TEST(Unit, MoveCanBeCalledMultipleTimesAndTranslationAccumulatesInOrder) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    system::Unit<double> u(geometry_op, system::SyncOperator<double> {});

    const Vector3<double> d1(1.0, 0.0, 0.0);
    const Vector3<double> d2(0.0, -2.0, 0.0);
    const Vector3<double> d3(0.5, 0.25, -3.0);

    u.move(d1);
    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(1.0, 0.0, 0.0),
        eps));

    u.move(d2);
    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(1.0, -2.0, 0.0),
        eps));

    u.move(d3);
    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(1.5, -1.75, -3.0),
        eps));
}

TEST(Unit, MoveWithZeroDeltaLeavesTranslationUnchanged) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> t0(-3.0, 4.0, 5.0);
    const math::Quaternion<double> q0(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    u.move(Vector3<double>(0.0, 0.0, 0.0));

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t0, eps));
}

TEST(Unit, MoveDoesNotModifyOrientation) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double angle = M_PI / 3.0;
    const math::Quaternion<double> q0(axis, angle);
    const Vector3<double> t0(1.0, 2.0, 3.0);

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const auto before = u.sync_operator();

    u.move(Vector3<double>(-7.0, 8.0, 9.0));

    const auto after = u.sync_operator();

    EXPECT_NEAR(after.orientation.w, before.orientation.w, eps);
    EXPECT_NEAR(after.orientation.x, before.orientation.x, eps);
    EXPECT_NEAR(after.orientation.y, before.orientation.y, eps);
    EXPECT_NEAR(after.orientation.z, before.orientation.z, eps);
}

TEST(Unit, MovePreservesRotationEffectOnDirections) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double angle = M_PI / 2.0;
    const math::Quaternion<double> q0(axis, angle);
    const Vector3<double> t0(0.0, 0.0, 0.0);

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_dir(1.0, 0.0, 0.0);
    const auto world_dir_before = u.sync_operator().sync_dir_to_world(local_dir);

    u.move(Vector3<double>(10.0, -3.0, 5.0));

    const auto world_dir_after = u.sync_operator().sync_dir_to_world(local_dir);

    EXPECT_TRUE(test::vec_near(world_dir_before, world_dir_after, eps));
    EXPECT_TRUE(test::vec_near(world_dir_after, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, MoveChangesWorldPointByExactlyDelta) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double angle = M_PI / 2.0;
    const math::Quaternion<double> q0(axis, angle);
    const Vector3<double> t0(1.0, 2.0, 3.0);

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_point(1.0, 0.0, 0.0);

    const auto world_before = u.sync_operator().sync_to_world(local_point);

    const Vector3<double> delta(-4.0, 5.0, 6.0);
    u.move(delta);

    const auto world_after = u.sync_operator().sync_to_world(local_point);

    EXPECT_TRUE(test::vec_near(world_after, world_before + delta, eps));
}

TEST(Unit, MoveAndInverseSyncRemainConsistentForPoints) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double angle = M_PI / 4.0;
    const math::Quaternion<double> q0(axis, angle);
    const Vector3<double> t0(-2.0, 1.0, 0.5);

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> local_point(2.0, -1.0, 4.0);

    u.move(Vector3<double>(3.0, -5.0, 7.0));

    const auto world_point = u.sync_operator().sync_to_world(local_point);
    const auto back_point  = u.sync_operator().sync_to_local(world_point);

    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));
}

TEST(Unit, MoveDoesNotAffectQueryAndGeometryOperatorAvailability) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    system::Unit<double> u(geometry_op, system::SyncOperator<double> {});

    u.move(Vector3<double>(1.0, 2.0, 3.0));

    const Vector3<double> query_point(0.25, -0.5, 0.75);

    const auto cp = u.geometry_operator().closest_point(query_point);
    const auto cn = u.geometry_operator().closest_normal(query_point);

    EXPECT_TRUE(test::is_finite_vec(cp));
    EXPECT_TRUE(test::is_finite_vec(cn));
}

TEST(Unit, ConsecutiveMoveForwardAndBackwardReturnsToOriginalTranslation) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    const Vector3<double> t0(2.0, -3.0, 4.0);
    const math::Quaternion<double> q0;

    system::Unit<double> u(geometry_op, test::make_sync_operator<double>(t0, q0));

    const Vector3<double> delta(5.5, -1.25, 2.75);

    u.move(delta);
    u.move(-delta);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t0, eps));
}