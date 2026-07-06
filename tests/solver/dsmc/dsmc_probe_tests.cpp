#include <atlas/solver/dsmc/dsmc_probe.h>

#include <gtest/gtest.h>

TEST(DsmcProbe, DefaultConstructsEmptyDeviceView) {
    const atlas::DsmcProbe probe;

    EXPECT_EQ(probe.velocity_ptr, nullptr);
    EXPECT_EQ(probe.internal_energy_ptr, nullptr);
    EXPECT_EQ(probe.species_ptr, nullptr);
    EXPECT_EQ(probe.properties_ptr, nullptr);
    EXPECT_EQ(probe.number_particle_ptr, nullptr);
    EXPECT_EQ(probe.max_relative_speed_ptr, nullptr);
    EXPECT_EQ(probe.max_sigma_g_ptr, nullptr);
    EXPECT_EQ(probe.collision_remainder_ptr, nullptr);
    EXPECT_EQ(probe.collision_count_ptr, nullptr);
    EXPECT_EQ(probe.indices_ptr, nullptr);
    EXPECT_EQ(probe.cell_start_ptr, nullptr);
    EXPECT_EQ(probe.cell_end_ptr, nullptr);
    EXPECT_EQ(probe.universe_volume_ptr, nullptr);
    EXPECT_EQ(probe.particle_count, 0);
    EXPECT_EQ(probe.species_count, 0);
    EXPECT_EQ(probe.cell_count, 0);
    EXPECT_EQ(probe.cell_volume, 0.0f);
    EXPECT_EQ(probe.statistical_weight, 0.0f);
    EXPECT_EQ(probe.kernel.type, atlas::DsmcKernelType::hard_sphere);
    EXPECT_EQ(probe.collision_seed, 0u);
}
