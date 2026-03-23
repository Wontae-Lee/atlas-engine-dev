#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Sync, DefaultConstructorBuildsIdentityPoseAndActsAsNoOp) {

    const system::Sync<double> s;

    const Vector3<double> point(1.0, -2.0, 3.5);

    const Vector3<double> dir(0.5, 2.0, -4.0);

    EXPECT_TRUE(test::vec_near(s.sync_to_world(point), point, eps));
    EXPECT_TRUE(test::vec_near(s.sync_to_local(point), point, eps));

    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(dir), dir, eps));
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_local(dir), dir, eps));

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

    const Vector3<double> t(-4.0, 5.0, 6.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::Sync<double> s(t, q);

    const Vector3<double> local_dir(1.0, 0.0, 0.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    const auto world_dir = s.sync_dir_to_world(local_dir);
    EXPECT_TRUE(test::vec_near(world_dir, expected_dir, eps));

    const auto back_dir = s.sync_dir_to_local(world_dir);
    EXPECT_TRUE(test::vec_near(back_dir, local_dir, eps));

    const Vector3<double> local_point(1.0, 0.0, 0.0);
    const Vector3<double> expected_point(-4.0, 6.0, 6.0);

    const auto world_point = s.sync_to_world(local_point);
    EXPECT_TRUE(test::vec_near(world_point, expected_point, eps));

    const auto back_point = s.sync_to_local(world_point);
    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));
}

TEST(Sync, RayTransformMatchesRigidRulesOriginRotatesAndTranslatesDirectionRotatesOnly) {

    const Vector3<double> t(10.0, -2.0, 3.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::Sync<double> s(t, q);

    const Ray<double> local(
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0));

    const Vector3<double> expected_origin(10.0, -1.0, 3.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    const auto world = s.sync_to_world(local);

    EXPECT_TRUE(test::vec_near(world.origin, expected_origin, eps));
    EXPECT_TRUE(test::vec_near(world.direction, expected_dir, eps));

    const auto back = s.sync_to_local(world);

    EXPECT_TRUE(test::vec_near(back.origin, local.origin, eps));
    EXPECT_TRUE(test::vec_near(back.direction, local.direction, eps));
}

TEST(Sync, SyncOperatorExposureIsConsistentWithWrapperResults) {

    const Vector3<double> t(1.0, 2.0, 3.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::Sync<double> s(t, q);

    const Vector3<double> p_local(1.0, 0.0, 0.0);
    const Vector3<double> d_local(1.0, 0.0, 0.0);

    const auto p_world_a = s.sync_to_world(p_local);
    const auto d_world_a = s.sync_dir_to_world(d_local);

    Vector3<double> p_world_b;
    Vector3<double> d_world_b;

    s.sync().sync_to_world(p_local, p_world_b);
    s.sync().sync_dir_to_world(d_local, d_world_b);

    EXPECT_TRUE(test::vec_near(p_world_a, p_world_b, eps));
    EXPECT_TRUE(test::vec_near(d_world_a, d_world_b, eps));
}

TEST(Sync, RebuildMatricesIsIdempotentAndDoesNotChangeResults) {

    const Vector3<double> t(3.0, -4.0, 5.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    system::Sync<double> s(t, q);

    const Vector3<double> p_local(1.0, 2.0, 3.0);
    const Vector3<double> d_local(1.0, -1.0, 0.5);
    const Ray<double> r_local(
        Vector3<double>(-2.0, 0.25, 1.5),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto p_world_0 = s.sync_to_world(p_local);
    const auto p_back_0  = s.sync_to_local(p_world_0);

    const auto d_world_0 = s.sync_dir_to_world(d_local);
    const auto d_back_0  = s.sync_dir_to_local(d_world_0);

    const auto r_world_0 = s.sync_to_world(r_local);
    const auto r_back_0  = s.sync_to_local(r_world_0);

    EXPECT_TRUE(test::vec_near(p_back_0, p_local, eps));
    EXPECT_TRUE(test::vec_near(d_back_0, d_local, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.origin, r_local.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.direction, r_local.direction, eps));

    s.rebuild_matrices();

    const auto p_world_1 = s.sync_to_world(p_local);
    const auto p_back_1  = s.sync_to_local(p_world_1);

    const auto d_world_1 = s.sync_dir_to_world(d_local);
    const auto d_back_1  = s.sync_dir_to_local(d_world_1);

    const auto r_world_1 = s.sync_to_world(r_local);
    const auto r_back_1  = s.sync_to_local(r_world_1);

    EXPECT_TRUE(test::vec_near(p_world_1, p_world_0, eps));
    EXPECT_TRUE(test::vec_near(p_back_1, p_back_0, eps));

    EXPECT_TRUE(test::vec_near(d_world_1, d_world_0, eps));
    EXPECT_TRUE(test::vec_near(d_back_1, d_back_0, eps));

    EXPECT_TRUE(test::vec_near(r_world_1.origin, r_world_0.origin, eps));
    EXPECT_TRUE(test::vec_near(r_world_1.direction, r_world_0.direction, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.origin, r_back_0.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.direction, r_back_0.direction, eps));

    EXPECT_TRUE(test::vec_near(p_back_1, p_local, eps));
    EXPECT_TRUE(test::vec_near(d_back_1, d_local, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.origin, r_local.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_1.direction, r_local.direction, eps));
}

TEST(Sync, SetOrientationRebuildsAndUpdatesRotation) {

    const Vector3<double> t(1.0, 2.0, 3.0);

    const math::Quaternion<double> q_identity;

    system::Sync<double> s(t, q_identity);

    const Vector3<double> x(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(x), x, eps));

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q_rot(axis, radians);

    s.set_orientation(q_rot);

    const Vector3<double> expected_y(0.0, 1.0, 0.0);
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_world(x), expected_y, eps));
    EXPECT_TRUE(test::vec_near(s.sync_dir_to_local(expected_y), x, eps));
}

TEST(Sync, SetPoseRebuildsAndUpdatesBothTranslationAndRotation) {

    const Vector3<double> t0(0.0, 0.0, 0.0);
    const math::Quaternion<double> q0;

    system::Sync<double> s(t0, q0);

    const Vector3<double> t_new(-4.0, 5.0, 6.0);

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q_new(axis, radians);

    s.set_pose(t_new, q_new);

    const Vector3<double> local_point(1.0, 0.0, 0.0);

    const Vector3<double> expected_ccw(-4.0, 6.0, 6.0);
    const Vector3<double> expected_cw(-4.0, 4.0, 6.0);

    const auto got = s.sync_to_world(local_point);

    EXPECT_TRUE(
        test::vec_near(got, expected_ccw, eps) || test::vec_near(got, expected_cw, eps));

    EXPECT_TRUE(test::vec_near(s.sync_to_local(got), local_point, eps));
}