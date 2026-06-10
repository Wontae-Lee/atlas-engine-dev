#include "../utilities/test_utils.h"

#include <atlas/collider/collider_probe.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using ColliderProbe = atlas::system::ColliderProbe<float>;

static_assert(std::is_same_v<atlas::ColliderProbe<float>, ColliderProbe>);

} // namespace

TEST(ColliderProbe, DefaultStateIsEmpty) {
    const ColliderProbe probe;

    EXPECT_EQ(probe.units, nullptr);
    EXPECT_EQ(probe.surface_interactions, nullptr);
    EXPECT_EQ(probe.flips, nullptr);
    EXPECT_EQ(probe.positions, nullptr);
    EXPECT_EQ(probe.velocities, nullptr);
    EXPECT_EQ(probe.internal_energies, nullptr);
    EXPECT_EQ(probe.species, nullptr);
    EXPECT_EQ(probe.materials, nullptr);
    EXPECT_EQ(probe.unit_count, 0);
    EXPECT_EQ(probe.interaction_count, 0);
    EXPECT_EQ(probe.flip_count, 0);
    EXPECT_EQ(probe.material_count, 0);
    EXPECT_EQ(probe.particle_count, 0);
}

TEST(ColliderProbe, PublicAliasMatchesSystemType) {
    EXPECT_TRUE((std::is_same_v<atlas::ColliderProbe<float>, ColliderProbe>));
}
