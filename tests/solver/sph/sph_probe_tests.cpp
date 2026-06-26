#include "../../utilities/test_utils.h"

#include <atlas/solver/sph/sph_probe.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using atlas::SphProbe;
using atlas::SphKernelType;
using atlas::SphSolverProbe;

} // namespace

TEST(SphProbe, DefaultConstructsEmptyDeviceView) {
    // Arrange and act: create the default probe.
    const SphProbe<float> probe;

    // Assert: pointer fields are null and scalar fields are empty.
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
    EXPECT_EQ(probe.num_of_cells, 0);
    EXPECT_EQ(probe.num_of_properties, 0);
    EXPECT_EQ(probe.kernel.type, SphKernelType::standard);
}

TEST(SphProbe, SolverProbeAliasMatchesProbeType) {
    // Assert: the public solver probe spelling aliases the probe type.
    EXPECT_TRUE((std::is_same_v<SphSolverProbe<float>, SphProbe<float>>));
}
