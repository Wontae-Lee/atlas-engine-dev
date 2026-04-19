#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/sync/sync.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Quat = atlas::Quaternion<T>;

constexpr T kEps = static_cast<T>(1e-4);

} // namespace

TEST(Sync, DefaultConstructorCreatesIdentityTransform) {
    const atlas::Sync<T> sync;

    EXPECT_TRUE(atlas::test::vec_near(sync.sync_to_world(Vec3(1, 2, 3)), Vec3(1, 2, 3), kEps));
}

TEST(Sync, BuilderConstructsConfiguredSync) {
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(Vec3(1, 2, 3), Quat::from_axis_angle(Vec3(0, 0, 1), T(atlas::pi / 2)))
                          .build();

    EXPECT_TRUE(atlas::test::vec_near(sync.sync_to_world(Vec3(1, 0, 0)), Vec3(1, 3, 3), 1e-3f));
}

TEST(Sync, BuilderRejectsInvalidQuaternion) {
    EXPECT_THROW(
        atlas::Sync<T>::builder()
            .with_rigid_pose(Vec3(0, 0, 0), Quat(T(0), T(0), T(0), T(0)))
            .build(),
        std::runtime_error);
}

TEST(Sync, SettersUpdatePose) {
    atlas::Sync<T> sync;

    sync.set_translation(Vec3(1, 2, 3));
    sync.set_orientation(Quat::from_axis_angle(Vec3(0, 0, 1), T(atlas::pi / 2)));

    EXPECT_TRUE(atlas::test::vec_near(sync.sync_to_world(Vec3(1, 0, 0)), Vec3(1, 3, 3), 1e-3f));
}

TEST(Sync, MakeHostSharedBuildsSync) {
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(Vec3(0, 0, 0), Quat(T(1), T(0), T(0), T(0)))
                          .make_host_shared();

    ASSERT_NE(sync, nullptr);
    EXPECT_TRUE(atlas::test::vec_near(sync->sync_to_world(Vec3(1, 0, 0)), Vec3(1, 0, 0), kEps));
}
