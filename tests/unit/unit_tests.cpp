#include "../utilities/tests_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Quat = atlas::Quaternion<T>;

atlas::GeometryHostPtr<T>
make_geometry() {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(Vec3(-1, -1, -1))
        .with_upper_corner(Vec3(1, 1, 1))
        .make_host_shared();
}

atlas::SyncHostPtr<T>
make_sync() {
    return atlas::Sync<T>::builder()
        .with_rigid_pose(Vec3(0, 0, 0), Quat(T(1), T(0), T(0), T(0)))
        .make_host_shared();
}

} // namespace

TEST(Unit, BuilderConstructsStaticUnit) {
    const auto unit = atlas::Unit<T>::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .build();

    EXPECT_FALSE(unit.dynamic());
}

TEST(Unit, BuilderConstructsDynamicUnitAndCanonicalizesKinematics) {
    const auto unit = atlas::Unit<T>::builder()
                          .with_geometry(make_geometry())
                          .with_sync(make_sync())
                          .with_velocity(Vec3(1, 0, 0))
                          .build();

    EXPECT_TRUE(unit.dynamic());
    ASSERT_TRUE(unit.velocity().has_value());
    ASSERT_TRUE(unit.acceleration().has_value());
    EXPECT_TRUE(atlas::test::vec_near(*unit.acceleration(), Vec3(0, 0, 0), 1e-6f));
}

TEST(Unit, BuilderRejectsMissingDependencies) {
    EXPECT_THROW(
        atlas::Unit<T>::builder()
            .with_sync(make_sync())
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Unit<T>::builder()
            .with_geometry(make_geometry())
            .build(),
        std::runtime_error);
}

TEST(Unit, UpdateAppliesVelocity) {
    auto unit = atlas::Unit<T>::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .with_velocity(Vec3(1, 0, 0))
                    .build();

    unit.update(1.0f);

    EXPECT_TRUE(atlas::test::vec_near(unit.sync_operator().translation, Vec3(1, 0, 0), 1e-4f));
}

TEST(Unit, MoveAndRotateUpdatePose) {
    auto unit = atlas::Unit<T>::builder()
                    .with_geometry(make_geometry())
                    .with_sync(make_sync())
                    .build();

    unit.move(Vec3(1, 2, 3));
    unit.rotate(Vec3(0, 0, 1), T(atlas::pi / 2));

    EXPECT_TRUE(atlas::test::vec_near(unit.sync_operator().translation, Vec3(1, 2, 3), 1e-4f));
    EXPECT_TRUE(atlas::test::vec_near(unit.sync_operator().sync_dir_to_world(Vec3(1, 0, 0)), Vec3(0, 1, 0), 1e-3f));
}
