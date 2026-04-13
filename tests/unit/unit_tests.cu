#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

#include <cmath>
#include <optional>
#include <stdexcept>

using namespace atlas;

CUDA_TEST(Unit, ConstructorStoresOperatorsAndLeavesKinematicsEmpty) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();
    const auto sop         = system::SyncOperator<double> {};

    system::Unit<double> u(geometry_op, sop);

    CUDA_EXPECT_FALSE(u.velocity().has_value());
    CUDA_EXPECT_FALSE(u.acceleration().has_value());
    CUDA_EXPECT_FALSE(u.angular_velocity().has_value());
    CUDA_EXPECT_FALSE(u.angular_acceleration().has_value());
    CUDA_EXPECT_FALSE(u.dynamic());

    CUDA_EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(0.0, 0.0, 0.0),
        eps));
}

CUDA_TEST(Unit, ConstructorWithKinematicsStoresAllValues) {
    const auto sphere      = test::make_sphere();
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

    CUDA_ASSERT_TRUE(u.velocity().has_value());
    CUDA_ASSERT_TRUE(u.acceleration().has_value());
    CUDA_ASSERT_TRUE(u.angular_velocity().has_value());
    CUDA_ASSERT_TRUE(u.angular_acceleration().has_value());

    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), v, eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.acceleration(), a, eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_velocity(), w, eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), alpha, eps));
    CUDA_EXPECT_TRUE(u.dynamic());
}

CUDA_TEST(Unit, SetGeometryOperatorReplacesStoredGeometryOperator) {
    const geometry::Sphere<double> sphere1(Vector3<double>(0.0, 0.0, 0.0), 1.0);
    const geometry::Sphere<double> sphere2(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    const auto geometry_op1 = sphere1.make_geometry_operator();
    const auto geometry_op2 = sphere2.make_geometry_operator();

    system::Unit<double> u(geometry_op1, system::SyncOperator<double> {});

    const Vector3<double> p(3.0, 0.0, 0.0);

    const auto cp_before = u.geometry_operator().closest_point(p);
    u.set_geometry_operator(geometry_op2);
    const auto cp_after = u.geometry_operator().closest_point(p);

    CUDA_EXPECT_FALSE(test::vec_near(cp_before, cp_after, eps));
    CUDA_EXPECT_TRUE(test::vec_near(cp_before, Vector3<double>(1.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(cp_after, Vector3<double>(2.0, 0.0, 0.0), eps));
}

CUDA_TEST(Unit, SetSyncOperatorReplacesStoredPose) {
    const auto sphere      = test::make_sphere();
    const auto geometry_op = sphere.make_geometry_operator();

    system::Unit<double> u(geometry_op, system::SyncOperator<double> {});

    const Vector3<double> t(1.0, 2.0, 3.0);
    const math::Quaternion<double> q(Vector3<double>(0.0, 0.0, 1.0), M_PI / 2.0);
    const system::SyncOperator<double> sop(t, q);

    u.set_sync_operator(sop);

    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t, eps));

    const auto dir = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    CUDA_EXPECT_TRUE(test::vec_near(dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

CUDA_TEST(Unit, SetGeometryOperatorAndSyncOperatorReplaceStoredState) {
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

    CUDA_EXPECT_TRUE(test::vec_near(cp, Vector3<double>(2.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, t, eps));
}

CUDA_TEST(Unit, DynamicIsFalseWhenNoKinematicStateExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {});

    CUDA_EXPECT_FALSE(u.dynamic());
}

CUDA_TEST(Unit, DynamicIsTrueWhenVelocityExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt,
        std::nullopt);

    CUDA_EXPECT_TRUE(u.dynamic());
}

CUDA_TEST(Unit, ConstructorAddsZeroVelocityWhenOnlyAccelerationExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt);

    CUDA_ASSERT_TRUE(u.velocity().has_value());
    CUDA_ASSERT_TRUE(u.acceleration().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(0.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.acceleration(), Vector3<double>(1.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(u.dynamic());
}

CUDA_TEST(Unit, DynamicIsTrueWhenAngularVelocityExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        std::nullopt);

    CUDA_EXPECT_TRUE(u.dynamic());
}

CUDA_TEST(Unit, ConstructorAddsZeroAngularVelocityWhenOnlyAngularAccelerationExists) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0));

    CUDA_ASSERT_TRUE(u.angular_velocity().has_value());
    CUDA_ASSERT_TRUE(u.angular_acceleration().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), Vector3<double>(0.0, 0.0, 1.0), eps));
    CUDA_EXPECT_TRUE(u.dynamic());
}

CUDA_TEST(Unit, UpdateDoesNothingWhenDtIsZero) {
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

    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), before_velocity, eps));
}

CUDA_TEST(Unit, UpdateDoesNothingWhenDtIsNegative) {
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

    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), before_velocity, eps));
}

CUDA_TEST(Unit, UpdateDoesNothingWhenUnitIsNotDynamic) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {});

    const auto before_translation = u.sync_operator().translation;
    const auto before_orientation = u.sync_operator().orientation;

    u.update(1.0);

    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, before_translation, eps));
    CUDA_EXPECT_NEAR(u.sync_operator().orientation.w, before_orientation.w, eps);
    CUDA_EXPECT_NEAR(u.sync_operator().orientation.x, before_orientation.x, eps);
    CUDA_EXPECT_NEAR(u.sync_operator().orientation.y, before_orientation.y, eps);
    CUDA_EXPECT_NEAR(u.sync_operator().orientation.z, before_orientation.z, eps);
}

CUDA_TEST(Unit, UpdateWithVelocityOnlyMovesByVelocityTimesDt) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, -2.0, 3.0),
        std::nullopt,
        std::nullopt,
        std::nullopt);

    u.update(0.5);

    CUDA_EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(0.5, -1.0, 1.5),
        eps));

    CUDA_ASSERT_TRUE(u.velocity().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(1.0, -2.0, 3.0), eps));
}

CUDA_TEST(Unit, UpdateWithVelocityAndAccelerationUpdatesVelocityThenMoves) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0),
        std::nullopt,
        std::nullopt);

    u.update(0.5);

    CUDA_ASSERT_TRUE(u.velocity().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(2.0, 0.0, 0.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(u.sync_operator().translation, Vector3<double>(1.0, 0.0, 0.0), eps));
}

CUDA_TEST(Unit, UpdateWithAngularVelocityOnlyRotates) {
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
    CUDA_EXPECT_TRUE(test::vec_near(world_dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

CUDA_TEST(Unit, UpdateWithAngularVelocityAndAngularAccelerationUpdatesAngularVelocityThenRotates) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        std::nullopt,
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        Vector3<double>(0.0, 0.0, 1.0));

    u.update(1.0);

    CUDA_ASSERT_TRUE(u.angular_velocity().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 2.0), eps));

    const auto expected = math::Quaternion<double>(Vector3<double>(0.0, 0.0, 1.0), 2.0).to_matrix3x3()
        * Vector3<double>(1.0, 0.0, 0.0);

    const auto got = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    CUDA_EXPECT_TRUE(test::vec_near(got, expected, eps));
}

CUDA_TEST(Unit, UpdateWithAngularVelocityVectorUsesItsDirectionAsRotationAxis) {
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

    CUDA_EXPECT_TRUE(
        test::vec_near(world_dir, expected_a, eps) || test::vec_near(world_dir, expected_b, eps));
}

CUDA_TEST(Unit, UpdateWithLinearAndAngularStateAppliesBoth) {
    const auto sphere = test::make_sphere();

    system::Unit<double> u(
        sphere.make_geometry_operator(),
        system::SyncOperator<double> {},
        Vector3<double>(1.0, 0.0, 0.0),
        std::nullopt,
        Vector3<double>(0.0, 0.0, 1.0),
        std::nullopt);

    u.update(M_PI / 2.0);

    CUDA_EXPECT_TRUE(test::vec_near(
        u.sync_operator().translation,
        Vector3<double>(M_PI / 2.0, 0.0, 0.0),
        eps));

    const auto dir = u.sync_operator().sync_dir_to_world(Vector3<double>(1.0, 0.0, 0.0));
    CUDA_EXPECT_TRUE(test::vec_near(dir, Vector3<double>(0.0, 1.0, 0.0), eps));
}

CUDA_TEST(Unit, BuilderBuildCreatesValidStaticUnit) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .build();

    CUDA_EXPECT_FALSE(u.velocity().has_value());
    CUDA_EXPECT_FALSE(u.acceleration().has_value());
    CUDA_EXPECT_FALSE(u.angular_velocity().has_value());
    CUDA_EXPECT_FALSE(u.angular_acceleration().has_value());
    CUDA_EXPECT_FALSE(u.dynamic());
}

CUDA_TEST(Unit, BuilderRetainsGeometryOwnerAfterSourcePointerIsReleased) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(
        geometry::Sphere<double>(Vector3<double>(0.0, 0.0, 0.0), 2.0));
    auto sync = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .build();

    geometry.reset();

    const auto cp = u.geometry_operator().closest_point(Vector3<double>(3.0, 0.0, 0.0));
    CUDA_EXPECT_TRUE(test::vec_near(cp, Vector3<double>(2.0, 0.0, 0.0), eps));
}

CUDA_TEST(Unit, BuilderAddsZeroAccelerationWhenVelocityExistsWithoutAcceleration) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .with_velocity(Vector3<double>(1.0, 2.0, 3.0))
                 .build();

    CUDA_ASSERT_TRUE(u.velocity().has_value());
    CUDA_ASSERT_TRUE(u.acceleration().has_value());

    CUDA_EXPECT_TRUE(test::vec_near(*u.velocity(), Vector3<double>(1.0, 2.0, 3.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.acceleration(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

CUDA_TEST(Unit, BuilderAddsZeroAngularAccelerationWhenAngularVelocityExistsWithoutAngularAcceleration) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto u = system::Unit<double>::builder()
                 .with_geometry(geometry)
                 .with_sync(sync)
                 .with_angular_velocity(Vector3<double>(0.0, 0.0, 2.0))
                 .build();

    CUDA_ASSERT_TRUE(u.angular_velocity().has_value());
    CUDA_ASSERT_TRUE(u.angular_acceleration().has_value());

    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_velocity(), Vector3<double>(0.0, 0.0, 2.0), eps));
    CUDA_EXPECT_TRUE(test::vec_near(*u.angular_acceleration(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

CUDA_TEST(Unit, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    auto ptr = system::Unit<double>::builder()
                   .with_geometry(geometry)
                   .with_sync(sync)
                   .make_host_shared();

    CUDA_ASSERT_TRUE(ptr != nullptr);
    CUDA_EXPECT_FALSE(ptr->dynamic());
}

CUDA_TEST(Unit, BuilderThrowsWhenGeometryIsNull) {
    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(nullptr),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderThrowsWhenSyncIsNull) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());

    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(geometry).with_sync(nullptr),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderThrowsWhenBuildWithoutGeometry) {
    auto sync = atlas::make_host_shared<system::Sync<double>>();

    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder().with_sync(sync).build(),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderThrowsWhenBuildWithoutSync) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());

    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder().with_geometry(geometry).build(),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderThrowsWhenAccelerationExistsWithoutVelocity) {
    auto geometry   = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    const auto sync = atlas::make_host_shared<system::Sync<double>>();

    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder()
            .with_geometry(geometry)
            .with_sync(sync)
            .with_acceleration(Vector3<double>(1.0, 0.0, 0.0))
            .build(),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderThrowsWhenAngularAccelerationExistsWithoutAngularVelocity) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    auto sync     = atlas::make_host_shared<system::Sync<double>>();

    CUDA_EXPECT_THROW(
        (void)system::Unit<double>::builder()
            .with_geometry(geometry)
            .with_sync(sync)
            .with_angular_acceleration(Vector3<double>(0.0, 0.0, 1.0))
            .build(),
        std::runtime_error);
}

CUDA_TEST(Unit, BuilderCanBeReusedAfterBuildBecauseStateIsReset) {
    auto geometry   = atlas::make_host_shared<geometry::Sphere<double>>(test::make_sphere());
    const auto sync = atlas::make_host_shared<system::Sync<double>>();

    auto builder = system::Unit<double>::builder();
    builder.with_geometry(geometry)
        .with_sync(sync)
        .with_velocity(Vector3<double>(1.0, 0.0, 0.0));

    auto u1 = builder.build();

    CUDA_EXPECT_TRUE(u1.velocity().has_value());
    CUDA_EXPECT_TRUE(test::vec_near(*u1.velocity(), Vector3<double>(1.0, 0.0, 0.0), eps));

    builder.with_geometry(geometry).with_sync(sync);

    auto u2 = builder.build();

    CUDA_EXPECT_FALSE(u2.velocity().has_value());
    CUDA_EXPECT_FALSE(u2.acceleration().has_value());
}
