#include "dsmc_test_utils.h"

#include <atlas/fluid/fluid_state.h>
#include <atlas/solver/dsmc/dsmc_energy_exchange_solver.h>

#include <testkit/testkit.h>

#include <cmath>
#include <stdexcept>

namespace {

using Solver = atlas::DsmcEnergyExchangeSolver<float>;

} // namespace

TEST(DsmcEnergyExchangeSolver, BuilderRejectsMissingDependencies) {
    EXPECT_THROW(Solver::builder().validate(), std::runtime_error);
}

TEST(DsmcEnergyExchangeSolver, BuilderCreatesConfiguredSolver) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);

    const auto solver = Solver::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_searcher(searcher)
        .with_kernel_type(atlas::DsmcKernelType::variable_hard_sphere)
        .with_workload_type(atlas::DsmcCollisionWorkloadType::flatten)
        .build();

    EXPECT_EQ(solver.kernel_type(), atlas::DsmcKernelType::variable_hard_sphere);
    EXPECT_EQ(solver.workload_type(), atlas::DsmcCollisionWorkloadType::flatten);
}

TEST(DsmcEnergyExchangeSolver, BuilderCreatesHostSharedSolver) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);

    const auto solver = Solver::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_searcher(searcher)
        .make_host_shared();

    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->kernel_type(), atlas::DsmcKernelType::hard_sphere);
}

TEST(DsmcEnergyExchangeSolver, ApplyCollisionAcceptsEmptyProbe) {
    Solver solver;

    EXPECT_NO_THROW(solver.apply_collision(nullptr, 0, 0.0f));
}

TEST(DsmcEnergyExchangeSolver, SampleUnitReturnsUnitIntervalValue) {
    const float sample = Solver::sample_unit(2, 5, 17u, 31u);

    EXPECT_GE(sample, 0.0f);
    EXPECT_LT(sample, 1.0f);
}

TEST(DsmcEnergyExchangeSolver, SampleBlRejectsInvalidExponents) {
    EXPECT_EQ(Solver::sample_bl(0.0f, 1.0f, 0, 0, 0u, 0u), 0.0f);
    EXPECT_EQ(Solver::sample_bl(1.0f, 0.0f, 0, 0, 0u, 0u), 0.0f);
}

TEST(DsmcEnergyExchangeSolver, SampleBlReturnsBoundedEnergyFraction) {
    const float sample = Solver::sample_bl(1.0f, 1.0f, 1, 3, 7u, 11u);

    EXPECT_GE(sample, 0.0f);
    EXPECT_LE(sample, 1.0f);
}

TEST(DsmcEnergyExchangeSolver, RelaxationProbabilityUsesMaterialSettings) {
    const auto material = atlas::MaterialProperties<float>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_rotational_dof(2)
        .with_vibrational_dof(2)
        .with_rotational_relaxation_probability(0.25f)
        .with_vibrational_relaxation_probability(0.5f)
        .build();

    EXPECT_EQ(Solver::rotational_relaxation_probability(material, 1.0f, 0.75f), 0.25f);
    EXPECT_EQ(Solver::vibrational_relaxation_probability(material, 1.0f, 0.75f), 0.5f);
}

TEST(DsmcEnergyExchangeSolver, ExchangeInternalEnergyReturnsTranslationalEnergyWithoutInternalState) {
    atlas::MaterialProperties<float> properties[1];
    properties[0] = atlas::test::make_dsmc_material();
    atlas::DsmcProbe<float> probe;
    probe.properties_ptr = properties;

    const float energy = Solver::exchange_internal_energy(
        probe,
        0,
        0,
        0,
        1,
        0,
        0,
        4.0f);

    EXPECT_NEAR(energy, 1.0f, 1.0e-6f);
}

TEST(DsmcEnergyExchangeSolver, ExchangeParticleInternalEnergyWritesModeSplit) {
    atlas::FluidInternalEnergy<float> energies[1];
    energies[0].rotational = 0.5f;
    energies[0].vibrational = 0.25f;

    atlas::DsmcProbe<float> probe;
    probe.internal_energy_ptr = energies;
    float disposable_energy = 1.0f;

    Solver::exchange_particle_internal_energy(
        probe,
        0,
        0,
        0,
        atlas::test::make_dsmc_material(),
        0.75f,
        3u,
        disposable_energy);

    EXPECT_EQ(energies[0].rotational, 0.0f);
    EXPECT_EQ(energies[0].vibrational, 0.0f);
    EXPECT_EQ(energies[0].translational, disposable_energy);
}

TEST(DsmcEnergyExchangeSolver, RescaleRelativeVelocityPreservesCenterVelocity) {
    const auto material = atlas::test::make_dsmc_material();
    atlas::Vector3F lhs(1.0f, 0.0f, 0.0f);
    atlas::Vector3F rhs(-1.0f, 0.0f, 0.0f);

    Solver::rescale_relative_velocity(lhs, rhs, material, material, 1.0f);

    EXPECT_NEAR(lhs.x + rhs.x, 0.0f, 1.0e-6f);
    EXPECT_NEAR(lhs.y + rhs.y, 0.0f, 1.0e-6f);
    EXPECT_NEAR(lhs.z + rhs.z, 0.0f, 1.0e-6f);
    EXPECT_TRUE(std::isfinite(lhs.length()));
    EXPECT_TRUE(std::isfinite(rhs.length()));
}

TEST(DsmcEnergyExchangeSolver, CollidePairRejectsInvalidParticleIndices) {
    atlas::DsmcProbe<float> probe;
    const int indices[] = { 0 };
    probe.indices_ptr = indices;
    probe.particle_count = 1;

    EXPECT_FALSE(Solver::collide_pair(probe, 0, 0, 0, 1, -1, 0, 1.0f));
}
