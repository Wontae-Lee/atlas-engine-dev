#include "../../utilities/test_utils.h"

#include <atlas/solver/hybrid/hybrid_dsmc_sph_probe.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using atlas::DsmcKernelType;
using atlas::HybridDsmcSphProbe;
using atlas::HybridProbe;
using atlas::SphKernelType;

} // namespace

TEST(HybridDsmcSphProbe, DefaultConstructsEmptyHybridProbe) {
    // Arrange and act: create the default hybrid probe.
    const HybridDsmcSphProbe<float> probe;

    // Assert: nested probes and hybrid routing fields start empty.
    EXPECT_EQ(probe.sph.kernel.type, SphKernelType::standard);
    EXPECT_EQ(probe.dsmc.kernel.type, DsmcKernelType::hard_sphere);
    EXPECT_EQ(probe.grouping_length, 0.0f);
    EXPECT_EQ(probe.sph_particle_threshold, 0);
    EXPECT_EQ(probe.collision_seed, 0u);
    EXPECT_FALSE(probe.pairing_without_replacement);
}

TEST(HybridDsmcSphProbe, PublicAliasMatchesProbeType) {
    // Assert: the public hybrid probe spelling aliases the concrete probe.
    EXPECT_TRUE((std::is_same_v<HybridProbe<float>, HybridDsmcSphProbe<float>>));
}
