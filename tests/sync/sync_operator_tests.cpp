#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SyncOperator, DefaultConstructorActsAsIdentityTransform) {

    const system::SyncOperator<double> op;

    const Vector3<double> point(1.0, -2.0, 3.5);
    const Vector3<double> dir(0.0, 2.0, -4.0);
    const Ray<double> ray(
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(0.0, 0.0, -2.0));

    Vector3<double> world_point;
    op.sync_to_world(point, world_point);
    EXPECT_TRUE(test::vec_near(world_point, point, eps));

    Vector3<double> local_point;
    op.sync_to_local(point, local_point);
    EXPECT_TRUE(test::vec_near(local_point, point, eps));

    Vector3<double> world_dir;
    op.sync_dir_to_world(dir, world_dir);
    EXPECT_TRUE(test::vec_near(world_dir, dir, eps));

    Vector3<double> local_dir;
    op.sync_dir_to_local(dir, local_dir);
    EXPECT_TRUE(test::vec_near(local_dir, dir, eps));

    Ray<double> world_ray;
    op.sync_to_world(ray, world_ray);
    EXPECT_TRUE(test::vec_near(world_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(world_ray.direction, ray.direction, eps));

    Ray<double> local_ray;
    op.sync_to_local(ray, local_ray);
    EXPECT_TRUE(test::vec_near(local_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(local_ray.direction, ray.direction, eps));
}

TEST(SyncOperator, ReturnByValueOverloadsActAsIdentityForDefaultPose) {

    const system::SyncOperator<double> op;

    const Vector3<double> point(1.0, -2.0, 3.0);
    const Vector3<double> dir(0.0, 2.0, 0.0);
    const Ray<double> ray(
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(0.0, 0.0, 1.0));

    const auto world_point = op.sync_to_world(point);
    const auto local_point = op.sync_to_local(point);
    EXPECT_TRUE(test::vec_near(world_point, point, eps));
    EXPECT_TRUE(test::vec_near(local_point, point, eps));

    const auto world_dir = op.sync_dir_to_world(dir);
    const auto local_dir = op.sync_dir_to_local(dir);
    EXPECT_TRUE(test::vec_near(world_dir, dir, eps));
    EXPECT_TRUE(test::vec_near(local_dir, dir, eps));

    const auto world_ray = op.sync_to_world(ray);
    const auto local_ray = op.sync_to_local(ray);
    EXPECT_TRUE(test::vec_near(world_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(world_ray.direction, ray.direction, eps));
    EXPECT_TRUE(test::vec_near(local_ray.origin, ray.origin, eps));
    EXPECT_TRUE(test::vec_near(local_ray.direction, ray.direction, eps));
}

TEST(SyncOperator, PureTranslationAffectsPointsButNotDirections) {

    const Vector3<double> t(5.0, -2.0, 1.0);
    const math::Quaternion<double> q;
    const system::SyncOperator<double> op(t, q);

    const Vector3<double> point(1.0, 2.0, 3.0);
    const Vector3<double> dir(0.0, 0.0, 1.0);

    Vector3<double> world_point;
    op.sync_to_world(point, world_point);
    EXPECT_TRUE(test::vec_near(world_point, point + t, eps));

    Vector3<double> local_point;
    op.sync_to_local(world_point, local_point);
    EXPECT_TRUE(test::vec_near(local_point, point, eps));

    Vector3<double> world_dir;
    op.sync_dir_to_world(dir, world_dir);
    EXPECT_TRUE(test::vec_near(world_dir, dir, eps));

    Vector3<double> local_dir;
    op.sync_dir_to_local(world_dir, local_dir);
    EXPECT_TRUE(test::vec_near(local_dir, dir, eps));
}

TEST(SyncOperator, RaySyncAppliesTranslationToOriginOnlyWhenRotationIsIdentity) {

    const Vector3<double> t(1.0, 2.0, 3.0);
    const math::Quaternion<double> q;
    const system::SyncOperator<double> op(t, q);

    const Ray<double> local(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0));

    Ray<double> world;
    op.sync_to_world(local, world);

    EXPECT_TRUE(test::vec_near(world.origin, t, eps));
    EXPECT_TRUE(test::vec_near(world.direction, local.direction, eps));

    Ray<double> back;
    op.sync_to_local(world, back);

    EXPECT_TRUE(test::vec_near(back.origin, local.origin, eps));
    EXPECT_TRUE(test::vec_near(back.direction, local.direction, eps));
}

TEST(SyncOperator, TranslationOrientationConstructorBuildsWorkingRigidPose) {

    const Vector3<double> t(-1.0, 2.0, -3.0);
    const math::Quaternion<double> q;
    const system::SyncOperator<double> op(t, q);

    const Vector3<double> point(2.0, 3.0, 4.0);

    const Vector3<double> world_point = op.sync_to_world(point);
    EXPECT_TRUE(test::vec_near(world_point, point + t, eps));

    EXPECT_TRUE(test::vec_near(op.sync_to_local(world_point), point, eps));
}

TEST(SyncOperator, RotationOnlyAffectsPointsAndDirections_NonIdentityQuaternion) {

    const Vector3<double> t(0.0, 0.0, 0.0);
    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::SyncOperator<double> op(t, q);

    const Vector3<double> local_point(1.0, 0.0, 0.0);
    const Vector3<double> local_dir(1.0, 0.0, 0.0);
    const Vector3<double> expected_world(0.0, 1.0, 0.0);

    Vector3<double> world_point;
    op.sync_to_world(local_point, world_point);
    EXPECT_TRUE(test::vec_near(world_point, expected_world, eps));

    Vector3<double> world_dir;
    op.sync_dir_to_world(local_dir, world_dir);
    EXPECT_TRUE(test::vec_near(world_dir, expected_world, eps));

    Vector3<double> back_point;
    op.sync_to_local(world_point, back_point);
    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));

    Vector3<double> back_dir;
    op.sync_dir_to_local(world_dir, back_dir);
    EXPECT_TRUE(test::vec_near(back_dir, local_dir, eps));
}

TEST(SyncOperator, RotationAndTranslationApplyCorrectlyToRay_NonIdentityQuaternion) {

    const Vector3<double> t(10.0, -2.0, 3.0);
    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::SyncOperator<double> op(t, q);

    const Ray<double> local(
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0));

    const Vector3<double> expected_origin(10.0, -1.0, 3.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    Ray<double> world;
    op.sync_to_world(local, world);

    EXPECT_TRUE(test::vec_near(world.origin, expected_origin, eps));
    EXPECT_TRUE(test::vec_near(world.direction, expected_dir, eps));

    Ray<double> back;
    op.sync_to_local(world, back);

    EXPECT_TRUE(test::vec_near(back.origin, local.origin, eps));
    EXPECT_TRUE(test::vec_near(back.direction, local.direction, eps));
}

TEST(SyncOperator, NonIdentityRotationWorksForReturnByValueOverloads) {

    const Vector3<double> t(-4.0, 5.0, 6.0);
    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    const system::SyncOperator<double> op(t, q);

    const Vector3<double> local_dir(1.0, 0.0, 0.0);
    const Vector3<double> expected_dir(0.0, 1.0, 0.0);

    const auto world_dir = op.sync_dir_to_world(local_dir);
    EXPECT_TRUE(test::vec_near(world_dir, expected_dir, eps));

    const auto back_dir = op.sync_dir_to_local(world_dir);
    EXPECT_TRUE(test::vec_near(back_dir, local_dir, eps));

    const Vector3<double> local_point(1.0, 0.0, 0.0);
    const Vector3<double> expected_point(-4.0, 6.0, 6.0);

    const auto world_point = op.sync_to_world(local_point);
    EXPECT_TRUE(test::vec_near(world_point, expected_point, eps));

    const auto back_point = op.sync_to_local(world_point);
    EXPECT_TRUE(test::vec_near(back_point, local_point, eps));
}

TEST(SyncOperator, NonUnitQuaternionBreaksRoundTripButNormalizedQuaternionRestoresIt) {

    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q_unit(axis, radians);

    math::Quaternion<double> q_non_unit = q_unit;
    q_non_unit *= 2.0;

    EXPECT_FALSE(std::abs(q_non_unit.length() - 1.0) < 1e-12);

    const Vector3<double> t(0.0, 0.0, 0.0);
    const system::SyncOperator<double> op_bad(t, q_non_unit);

    const Vector3<double> local_point(1.25, -0.5, 2.0);

    Vector3<double> p_world_bad;
    op_bad.sync_to_world(local_point, p_world_bad);

    Vector3<double> p_back_bad;
    op_bad.sync_to_local(p_world_bad, p_back_bad);

    EXPECT_FALSE(test::vec_near(p_back_bad, local_point, eps));

    const Vector3<double> local_dir(1.0, 2.0, -3.0);

    Vector3<double> d_world_bad;
    op_bad.sync_dir_to_world(local_dir, d_world_bad);

    const double len_local = std::sqrt(
        local_dir.x * local_dir.x
        + local_dir.y * local_dir.y
        + local_dir.z * local_dir.z);

    const double len_world_bad = std::sqrt(
        d_world_bad.x * d_world_bad.x
        + d_world_bad.y * d_world_bad.y
        + d_world_bad.z * d_world_bad.z);

    EXPECT_FALSE(std::abs(len_world_bad - len_local) < eps);

    math::Quaternion<double> q_fixed = q_non_unit.normalized();
    EXPECT_TRUE(std::abs(q_fixed.length() - 1.0) < 1e-12);

    const system::SyncOperator<double> op_good(t, q_fixed);

    Vector3<double> p_world_good;
    op_good.sync_to_world(local_point, p_world_good);

    Vector3<double> p_back_good;
    op_good.sync_to_local(p_world_good, p_back_good);

    EXPECT_TRUE(test::vec_near(p_back_good, local_point, eps));

    Vector3<double> d_world_good;
    op_good.sync_dir_to_world(local_dir, d_world_good);

    const double len_world_good = std::sqrt(
        d_world_good.x * d_world_good.x
        + d_world_good.y * d_world_good.y
        + d_world_good.z * d_world_good.z);

    EXPECT_TRUE(std::abs(len_world_good - len_local) < 1e-12);
}

TEST(SyncOperator, RebuildMatricesIsIdempotentAndKeepsTransformsConsistent) {

    const Vector3<double> t(3.0, -4.0, 5.0);
    const Vector3<double> axis(0.0, 0.0, 1.0);
    constexpr double radians = M_PI / 2.0;
    const math::Quaternion<double> q(axis, radians);

    system::SyncOperator<double> op(t, q);

    const Vector3<double> p_local(1.0, 2.0, 3.0);
    const Vector3<double> d_local(1.0, -1.0, 0.5);
    const Ray<double> r_local(
        Vector3<double>(-2.0, 0.25, 1.5),
        Vector3<double>(0.0, 1.0, 0.0));

    Vector3<double> p_world_0;
    Vector3<double> p_back_0;
    Vector3<double> d_world_0;
    Vector3<double> d_back_0;
    Ray<double> r_world_0;
    Ray<double> r_back_0;

    op.sync_to_world(p_local, p_world_0);
    op.sync_to_local(p_world_0, p_back_0);

    op.sync_dir_to_world(d_local, d_world_0);
    op.sync_dir_to_local(d_world_0, d_back_0);

    op.sync_to_world(r_local, r_world_0);
    op.sync_to_local(r_world_0, r_back_0);

    EXPECT_TRUE(test::vec_near(p_back_0, p_local, eps));
    EXPECT_TRUE(test::vec_near(d_back_0, d_local, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.origin, r_local.origin, eps));
    EXPECT_TRUE(test::vec_near(r_back_0.direction, r_local.direction, eps));

    op.rebuild_matrices();

    Vector3<double> p_world_1;
    Vector3<double> p_back_1;
    Vector3<double> d_world_1;
    Vector3<double> d_back_1;
    Ray<double> r_world_1;
    Ray<double> r_back_1;

    op.sync_to_world(p_local, p_world_1);
    op.sync_to_local(p_world_1, p_back_1);

    op.sync_dir_to_world(d_local, d_world_1);
    op.sync_dir_to_local(d_world_1, d_back_1);

    op.sync_to_world(r_local, r_world_1);
    op.sync_to_local(r_world_1, r_back_1);

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