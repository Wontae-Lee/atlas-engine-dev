#include <atlas/solver/sph/sph_probe.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::SphKernelType;
using atlas::SphProbe;
using atlas::SphSolverProbe;

}

TEST(SphProbe, DefaultConstructsEmptyDeviceView) {
    const SphProbe probe;

    EXPECT_EQ(probe.position_ptr, nullptr);
    EXPECT_EQ(probe.velocity_ptr, nullptr);
    EXPECT_EQ(probe.species_ptr, nullptr);
    EXPECT_EQ(probe.properties_ptr, nullptr);
    EXPECT_EQ(probe.number_particle_ptr, nullptr);
    EXPECT_EQ(probe.field_force_ptr, nullptr);
    EXPECT_EQ(probe.indices_ptr, nullptr);
    EXPECT_EQ(probe.cell_start_ptr, nullptr);
    EXPECT_EQ(probe.cell_end_ptr, nullptr);
    EXPECT_EQ(probe.neighbor_offsets_ptr, nullptr);
    EXPECT_EQ(probe.neighbor_indices_ptr, nullptr);
    EXPECT_EQ(probe.inverse_cell_size, 0.0f);
    EXPECT_EQ(probe.cell_size, 0.0f);
    EXPECT_EQ(probe.particle_count, 0);
    EXPECT_EQ(probe.cell_count, 0);
    EXPECT_EQ(probe.property_count, 0);
    EXPECT_EQ(probe.kernel.type, SphKernelType::standard);
}

TEST(SphProbe, SolverProbeAliasMatchesProbeType) {
    EXPECT_TRUE((std::is_same_v<SphSolverProbe, SphProbe>));
}
