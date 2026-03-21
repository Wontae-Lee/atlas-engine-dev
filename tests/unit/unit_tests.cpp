#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <stdexcept>

using namespace atlas;

TEST(Unit, ConstructorStoresOperatorsAndLeavesKinematicsEmpty) {
    const auto sphere = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();
    const auto sop         = system::SyncOperator<double> {};

    system::Unit<double> u(geometry_op, sop);

    EXPECT_FALSE(u.velocity().has_value());
    EXPECT_FALSE(u.acceleration().has_value());
    EXPECT_FALSE(u.angular_velocity().has_value());
    EXPECT_FALSE(u.angular_acceleration().has_value());
    EXPECT_FALSE(u.dynamic());

    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(0.0, 0.0, 0.0),
        eps));
}

TEST(Unit, ConstructorWithKinematicsStoresAllValues) {
    const auto sphere = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();
    const auto sop         = system::SyncOperator<double> {};

    const Vector3<double> v(1.0, 2.0, 3.0);
    const Vector3<double> a(-1.0, 0.5, 4.0);
    const Vector3<double> w(0.0, 0.0, 2.0);
    const Vector3<double> alpha(0.0, 0.0, -0.25);

    system::Unit<double> u(
        geometry_op,
        sop,
        v,
        a,
        w,
        alpha);

    ASSERT_TRUE(u.velocity().has_value());
    ASSERT_TRUE(u.acceleration().has_value());
    ASSERT_TRUE(u.angular_velocity().has_value());
    ASSERT_TRUE(u.angular_acceleration().has_value());

    EXPECT_TRUE(test::vec_near(*u.velocity(), v, eps));
    EXPECT_TRUE(test::vec_near(*u.acceleration(), a, eps));
    EXPECT_TRUE(test::vec_near(*u.angular_velocity(), w, eps));
    EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), alpha, eps));
    EXPECT_TRUE(u.dynamic());
}

TEST(Unit, SetGeometryOperatorReplacesStoredGeometryOperator) {
    const geometry::Sphere<double> sphere1(Vector3<double>(0.0, 0.0, 0.0), 1.0);
    const geometry::Sphere<double> sphere2(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    const auto geometry_op1 = sphere1.make_geometry_operator();
    const auto geometry_op2 = sphere2.make_geometry_operator();

    system::Unit<double> u(geometry_op1, system::SyncOperator<double> {});

    const Vector3<double> p(3.0, 0.0, 0.0);

    const auto cp_before = u.geometry_operator().closest_point(p);
    u.set_geometry_operator(geometry_op2);
    const auto cp_after = u.geometry_operator().closest_point(p);

    EXPECT_FALSE(test::vec_near(cp_before, cp_after, eps));
    EXPECT_TRUE(test::vec_near(cp_before, Vector3<double>(1.0, 0.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(cp_after, Vector3<double>(2.0, 0.0, 0.0), eps));
}

TEST(Unit, SetSyncOperatorReplacesStoredPose) {
    const auto sphere = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    system::Unit<double> u(geometry_op, system::SyncOperator<double> {});

    const Vector3<double> t(1.0, 2.0, 3.0);
    const math::Quaternion<double> q(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);
    const system::SyncOperator<double> sop(t, q);

    u.set_sync_operator(sop);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t, eps));

    const auto dir = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    EXPECT_TRUE(test::vec_near(dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, SetGeometryOperatorAndSyncOperatorReplaceStoredState) {
    const geometry::Sphere<double> sphere1(Vector3<double>(0.0, 0.0, 0.0), 1.0);
    const geometry::Sphere<double> sphere2(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    const auto geometry_op1 = sphere1.make_geometry_operator();
    const auto geometry_op2 = sphere2.make_geometry_operator();

    system::Unit<double> u(geometry_op1, system::SyncOperator<double> {});

    const Vector3<double> t(0.0, 1.0, 0.0);
    const math::Quaternion<double> q;
    const system::SyncOperator<double> sop2(t, q);

    u.set_geometry_operator(geometry_op2);
    u.set_sync_operator(sop2);

    const Vector3<double> p(3.0, 0.0, 0.0);
    const auto cp = u.geometry_operator().closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, Vector3<double>(2.0, 0.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t, eps));
}

TEST(Unit, DynamicIsFalseWhenNoKinematicStateExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {});

    EXPECT_FALSE(u.dynamic());
}

TEST(Unit, DynamicIsTrueWhenVelocityExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt,
        std::nullopt);

    EXPECT_TRUE(u.dynamic());
}

TEST(Unit, ConstructorAddsZeroVelocityWhenOnlyAccelerationExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt);

    ASSERT_TRUE(u.velocity().has_value());
    ASSERT_TRUE(u.acceleration().has_value());
    EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(0.0, 0.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(*u.acceleration(), Vector3<double>(1.0, 0.0, 0.0), eps));
    EXPECT_TRUE(u.dynamic());
}

TEST(Unit, DynamicIsTrueWhenAngularVelocityExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        std::nullopt);

    EXPECT_TRUE(u.dynamic());
}

TEST(Unit, ConstructorAddsZeroAngularVelocityWhenOnlyAngularAccelerationExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0));

    ASSERT_TRUE(u.angular_velocity().has_value());
    ASSERT_TRUE(u.angular_acceleration().has_value());
    EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), Vector3<double>(0.0, 0.0, 1.0), eps));
    EXPECT_TRUE(u.dynamic());
}

TEST(Unit, UpdateDoesNothingWhenDtIsZero) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(4.0, 5.0, 6.0),
        std::nullopt,
        std::nullopt);

    const auto before_translation = u.sync_operator().translation;
    const auto before_velocity    = *u.velocity();

    u.update(0.0);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    EXPECT_TRUE(test::vec_near(*u.velocity(), before_velocity, eps));
}

TEST(Unit, UpdateDoesNothingWhenDtIsNegative) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(4.0, 5.0, 6.0),
        std::nullopt,
        std::nullopt);

    const auto before_translation = u.sync_operator().translation;
    const auto before_velocity    = *u.velocity();

    u.update(-1.0);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    EXPECT_TRUE(test::vec_near(*u.velocity(), before_velocity, eps));
}

TEST(Unit, UpdateDoesNothingWhenUnitIsNotDynamic) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {});

    const auto before_translation = u.sync_operator().translation;
    const auto before_orientation = u.sync_operator().orientation;

    u.update(1.0);

    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    EXPECT_NEAR(u.sync_operator().orientation.w, before_orientation.w, eps);
    EXPECT_NEAR(u.sync_operator().orientation.x, before_orientation.x, eps);
    EXPECT_NEAR(u.sync_operator().orientation.y, before_orientation.y, eps);
    EXPECT_NEAR(u.sync_operator().orientation.z, before_orientation.z, eps);
}

TEST(Unit, UpdateWithVelocityOnlyMovesByVelocityTimesDt) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, -2.0, 3.0),
        std::nullopt,
        std::nullopt,
        std::nullopt);

    u.update(0.5);

    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(0.5, -1.0, 1.5),
        eps));

    ASSERT_TRUE(u.velocity().has_value());
    EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(1.0, -2.0, 3.0), eps));
}

TEST(Unit, UpdateWithVelocityAndAccelerationUpdatesVelocityThenMoves) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt);

    u.update(0.5);

    ASSERT_TRUE(u.velocity().has_value());
    EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(2.0, 0.0, 0.0), eps));
    EXPECT_TRUE(test::vec_near(u.sync_operator().translation, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(Unit, UpdateWithAngularVelocityOnlyRotates) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        std::nullopt);

    u.update(M_PI / 2.0);

    const auto world_dir = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    EXPECT_TRUE(test::vec_near(world_dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, UpdateWithAngularVelocityAndAngularAccelerationUpdatesAngularVelocityThenRotates) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        Vector3<double>(0.0, 0.0, 1.0));

    u.update(1.0);

    ASSERT_TRUE(u.angular_velocity().has_value());
    EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 2.0), eps));

    const auto expected = math::Quaternion<double>(Vector3<double>(0.0, 0.0, 1.0), 2.0).to_matrix3x3()
        * Vector3<double>(1.0, 0.0, 0.0);

    const auto got = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Unit, UpdateWithAngularVelocityVectorUsesItsDirectionAsRotationAxis) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 2.0, 0.0),
        std::nullopt);

    u.update(M_PI / 4.0);

    const auto world_dir = u.sync_operator().sync_dir_to_world(Vector3<double>(0.0, 0.0, 1.0));

    const Vector3<double> expected_a(1.0, 0.0, 0.0);
    const Vector3<double> expected_b(-1.0, 0.0, 0.0);

    EXPECT_TRUE(
        test::vec_near(world_dir, expected_a, eps) || test::vec_near(world_dir, expected_b, eps));
}

TEST(Unit, UpdateWithLinearAndAngularStateAppliesBoth) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        std::nullopt);

    u.update(M_PI / 2.0);

    EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(M_PI / 2.0, 0.0, 0.0),
        eps));

    const auto dir = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    EXPECT_TRUE(test::vec_near(dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

TEST(Unit, BuilderBuildCreatesValidStaticUnit) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .build();

    EXPECT_FALSE(u.velocity().has_value());
    EXPECT_FALSE(u.acceleration().has_value());
    EXPECT_FALSE(u.angular_velocity().has_value());
    EXPECT_FALSE(u.angular_acceleration().has_value());
    EXPECT_FALSE(u.dynamic());
}

TEST(Unit, BuilderRetainsGeometryOwnerAfterSourcePointerIsReleased) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(
        geometry::Sphere<double>(Vector3<double>(0.0, 0.0, 0.0), 2.0));
    auto sync = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .build();

    geometry.reset();

    const auto cp = u.geometry_operator().closest_point(Vector3<double>(3.0, 0.0, 0.0));
    EXPECT_TRUE(test::vec_near(cp, Vector3<double>(2.0, 0.0, 0.0), eps));
}

TEST(Unit, BuilderAddsZeroAccelerationWhenVelocityExistsWithoutAcceleration) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .with_velocity(Vector3<double>(1.0, 2.0, 3.0))
                 .build();

    ASSERT_TRUE(u.velocity().has_value());
    ASSERT_TRUE(u.acceleration().has_value());

    EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(1.0, 2.0, 3.0), eps));
    EXPECT_TRUE(test::vec_near(*u.acceleration(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(Unit, BuilderAddsZeroAngularAccelerationWhenAngularVelocityExistsWithoutAngularAcceleration) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .with_angular_velocity(Vector3<double>(0.0, 0.0, 2.0))
                 .build();

    ASSERT_TRUE(u.angular_velocity().has_value());
    ASSERT_TRUE(u.angular_acceleration().has_value());

    EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 2.0), eps));
    EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(Unit, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto ptr = system::Unit<double>::builder()
                   .with_geometry(geometry)
                   .with_sync(sync)
                   .make_host_shared();

    ASSERT_TRUE(ptr != nullptr);
    EXPECT_FALSE(ptr->dynamic());
}

TEST(Unit, BuilderThrowsWhenGeometryIsNull) {
    EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(nullptr),
        std::runtime_error);
}

TEST(Unit, BuilderThrowsWhenSyncIsNull) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());

    EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(geometry).with_sync(nullptr),
        std::runtime_error);
}

TEST(Unit, BuilderThrowsWhenBuildWithoutGeometry) {
    auto sync = atlas::make_host_shared<system::Sync<double>>();

    EXPECT_THROW(
        (void)system::Unit<double>::builder().with_sync(sync).build(),
        std::runtime_error);
}

TEST(Unit, BuilderThrowsWhenBuildWithoutSync) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());

    EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(geometry).build(),
        std::runtime_error);
}

TEST(Unit, BuilderThrowsWhenAccelerationExistsWithoutVelocity) {
    auto geometry   = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    const auto sync = atlas::make_host_shared<system::Sync<double>>();

    EXPECT_THROW(
        (void)system::Unit<double>::builder()
            .with_geometry(geometry)
            .with_sync(sync)
            .with_acceleration(Vector3<double>(1.0, 0.0, 0.0))
            .build(),
        std::runtime_error);
}

TEST(Unit, BuilderThrowsWhenAngularAccelerationExistsWithoutAngularVelocity) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    EXPECT_THROW(
        (void)system::Unit<double>::builder()
            .with_geometry(geometry)
            .with_sync(sync)
            .with_angular_acceleration(Vector3<double>(0.0, 0.0, 1.0))
            .build(),
        std::runtime_error);
}

TEST(Unit, BuilderCanBeReusedAfterBuildBecauseStateIsReset) {
    auto geometry   = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    const auto sync = atlas::make_host_shared<system::Sync<double>>();

    auto builder = system::Unit<double>::builder();
    builder.with_geometry(geometry)
        .with_sync(sync)
        .with_velocity(Vector3<double>(1.0, 0.0, 0.0));

    auto u1 = builder.build();

    EXPECT_TRUE(u1.velocity().has_value());
    EXPECT_TRUE(test::vec_near(*u1.velocity(), Vector3<double>(1.0, 0.0, 0.0), eps));

    builder.with_geometry(geometry).with_sync(sync);

    auto u2 = builder.build();

    EXPECT_FALSE(u2.velocity().has_value());
    EXPECT_FALSE(u2.acceleration().has_value());
}
