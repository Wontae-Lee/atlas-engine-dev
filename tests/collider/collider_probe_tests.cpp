#include <atlas/collider/collider_probe.h>

#include <gtest/gtest.h>

TEST(ColliderProbe, DefaultStateIsEmpty) {
    const atlas::ColliderProbe probe;

    EXPECT_EQ(probe.units, nullptr);
    EXPECT_EQ(probe.unit_bounds, nullptr);
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
    EXPECT_FALSE(probe.scene_bound_covers_units);
}
