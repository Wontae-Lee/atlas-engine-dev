#include <atlas/solver/dsmc/dsmc_solver.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/material/molecule.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver_type.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

using atlas::DsmcKernelType;
using atlas::DsmcSolver;
using atlas::SolverType;

struct CollisionFixture {
    atlas::FluidHostPtr fluid;
    atlas::UniverseHostPtr universe;
    atlas::SpatialHashingSearcherHostPtr searcher;
};

CollisionFixture
make_collision_fixture(const bool two_cells) {
    using namespace atlas;
    const std::size_t count = two_cells ? 6 : 3;
    HostBuffer<Float3> positions(count);
    HostBuffer<Float3> velocities(count);
    const Float3 velocity_pattern[] {
        Float3(300.0f, 0.0f, 0.0f),
        Float3(-200.0f, 100.0f, 0.0f),
        Float3(0.0f, -100.0f, 250.0f)
    };
    for (std::size_t i = 0; i < count; ++i) {
        positions[i] = Float3(0.2f + 0.1f * static_cast<float>(i % 3)
                                 + static_cast<float>(i / 3),
                             0.2f, 0.2f);
        velocities[i] = velocity_pattern[i % 3];
    }
    auto materials = MaterialDictionary::builder()
                         .with_material(Material(Molecule(4.65e-26f, 0.0f, 0.0f, 0.0f,
                                                          4.17e-10f, 273.0f, 0.74f, 1.0f)))
                         .make_host_shared();
    auto fluid = Fluid::builder()
                     .with_buffer_size(count)
                     .with_particle_count(count)
                     .with_statistical_weight(1.0e22f)
                     .with_materials(materials)
                     .make_host_unique();
    fluid->state<FluidPositionState>()->data()
        = DeviceBuffer<Float3>(positions.begin(), positions.end());
    fluid->state<FluidVelocityState>()->data()
        = DeviceBuffer<Float3>(velocities.begin(), velocities.end());
    auto universe = Universe::builder()
                        .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
                        .with_upper_corner(Float3(two_cells ? 1.9f : 0.9f, 0.9f, 0.9f))
                        .with_cell_size(1.0f)
                        .make_host_unique();
    const auto cells = static_cast<std::size_t>(universe->cell_count());
    universe->emplace_state<UniverseNumberParticleState>(cells);
    universe->emplace_state<UniverseMaxRelativeSpeedState>(cells);
    universe->emplace_state<UniverseMaxSigmaGState>(cells);
    universe->emplace_state<UniverseCollisionCountState>(cells);
    auto searcher = SpatialHashingSearcher::builder().with_universe(*universe).make_host_shared();
    searcher->classify(fluid->state<FluidPositionState>(),
                       universe->state<UniverseNumberParticleState>(),
                       static_cast<int>(count));
    return { std::move(fluid), std::move(universe), std::move(searcher) };
}

atlas::HostBuffer<atlas::Float3>
read_velocities(const atlas::Fluid& fluid) {
    const auto& velocities = fluid.state<atlas::FluidVelocityState>()->data();
    return { velocities.begin(), velocities.end() };
}

struct ConservedTotals {
    double momentum[3] {};
    double momentum_scale = 0.0;
    double energy = 0.0;
};

ConservedTotals
conserved_totals(const atlas::HostBuffer<atlas::Float3>& velocities) {
    ConservedTotals totals;
    constexpr double mass = 4.65e-26;
    for (const auto& velocity : velocities) {
        const double x = velocity.x;
        const double y = velocity.y;
        const double z = velocity.z;
        totals.momentum[0] += mass * x;
        totals.momentum[1] += mass * y;
        totals.momentum[2] += mass * z;
        const double speed_squared = x * x + y * y + z * z;
        totals.momentum_scale += mass * std::sqrt(speed_squared);
        totals.energy += 0.5 * mass * speed_squared;
    }
    return totals;
}

bool
velocities_changed(const atlas::HostBuffer<atlas::Float3>& before,
                   const atlas::HostBuffer<atlas::Float3>& after,
                   const std::size_t first, const std::size_t count) {
    for (std::size_t i = first; i < first + count; ++i) {
        if (before[i] != after[i]) return true;
    }
    return false;
}

}

TEST(DsmcSolver, DefaultBuilderYieldsDefaultParameters) {
    const DsmcSolver solver = DsmcSolver::builder().build();

    EXPECT_EQ(solver.type(), SolverType::dsmc);
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::hard_sphere);
    EXPECT_EQ(solver.majorant_sample_pairs(), 8);
    EXPECT_EQ(solver.majorant_exhaustive_limit(), 5);
}

TEST(DsmcSolver, DefaultConstructedSolverMatchesBuilderDefaults) {
    const DsmcSolver solver {};

    EXPECT_EQ(solver.type(), SolverType::dsmc);
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::hard_sphere);
    EXPECT_EQ(solver.majorant_sample_pairs(), 8);
    EXPECT_EQ(solver.majorant_exhaustive_limit(), 5);
}

TEST(DsmcSolver, BuilderAppliesConfiguredParameters) {
    const DsmcSolver solver = DsmcSolver::builder()
                                  .with_kernel_type(DsmcKernelType::variable_soft_sphere)
                                  .with_majorant_sample_pairs(16)
                                  .with_majorant_exhaustive_limit(4)
                                  .build();

    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_soft_sphere);
    EXPECT_EQ(solver.majorant_sample_pairs(), 16);
    EXPECT_EQ(solver.majorant_exhaustive_limit(), 4);
}

TEST(DsmcSolver, ExplicitConstructorStoresParameters) {
    const DsmcSolver solver(DsmcKernelType::variable_hard_sphere, 3, 7);

    EXPECT_EQ(solver.type(), SolverType::dsmc);
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_hard_sphere);
    EXPECT_EQ(solver.majorant_sample_pairs(), 3);
    EXPECT_EQ(solver.majorant_exhaustive_limit(), 7);
}

TEST(DsmcSolver, BuilderRejectsMajorantSamplePairsBelowOne) {
    // Estimating a large cell's majorant (sigma*g)_max needs at least one sampled pair; zero
    // samples would leave the NTC acceptance bound undefined.
    EXPECT_THROW(
        static_cast<void>(DsmcSolver::builder().with_majorant_sample_pairs(0).build()),
        std::runtime_error);
}

TEST(DsmcSolver, BuilderRejectsMajorantExhaustiveLimitBelowTwo) {
    // Below this occupancy the majorant is scanned exactly, and a collision pair needs two
    // particles, so a limit under two could never enclose one.
    EXPECT_THROW(
        static_cast<void>(DsmcSolver::builder().with_majorant_exhaustive_limit(1).build()),
        std::runtime_error);
}

TEST(DsmcSolver, MakeHostSharedProducesDsmcSolver) {
    const auto solver = DsmcSolver::builder()
                            .with_kernel_type(DsmcKernelType::variable_hard_sphere)
                            .make_host_shared();

    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->type(), SolverType::dsmc);
    EXPECT_EQ(solver->kernel_type(), DsmcKernelType::variable_hard_sphere);
}

TEST(DsmcSolver, OverlappingCandidatesConserveMomentumAndEnergy) {
    auto fixture = make_collision_fixture(false);
    ASSERT_EQ(fixture.universe->cell_count(), 1);
    const auto before = read_velocities(*fixture.fluid);
    const auto expected = conserved_totals(before);
    auto solver = DsmcSolver::builder()
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_majorant_exhaustive_limit(4)
                      .build();
    for (int step = 0; step < 5; ++step) {
        solver.solve(*fixture.fluid, *fixture.universe, fixture.searcher->view(), 0, 1.0e-4f);
        const auto& counts = fixture.universe->state<atlas::UniverseCollisionCountState>()->data();
        const atlas::HostBuffer<int> host_counts(counts.begin(), counts.end());
        ASSERT_GT(host_counts[0], 64) << "step " << step;
        const auto actual = conserved_totals(read_velocities(*fixture.fluid));
        EXPECT_NEAR(actual.energy, expected.energy, expected.energy * 1.0e-4) << "step " << step;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(actual.momentum[axis], expected.momentum[axis],
                        expected.momentum_scale * 1.0e-4)
                << "step " << step << ", axis " << axis;
        }
    }
    EXPECT_TRUE(velocities_changed(before, read_velocities(*fixture.fluid), 0, before.size()));
}

TEST(DsmcSolver, OverlappingCandidatesLeaveOtherSolversCellsUntouched) {
    using namespace atlas;
    auto fixture = make_collision_fixture(true);
    ASSERT_EQ(fixture.universe->cell_count(), 2);
    const HostBuffer<int> allocation { 0, 1 };
    fixture.universe->emplace_state<UniverseAllocatedSolverState>(
        DeviceBuffer<int>(allocation.begin(), allocation.end()));
    const HostBuffer<int> initial_counts { 0, 73 };
    fixture.universe->state<UniverseCollisionCountState>()->data()
        = DeviceBuffer<int>(initial_counts.begin(), initial_counts.end());
    const auto before = read_velocities(*fixture.fluid);
    auto solver = DsmcSolver::builder()
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_majorant_exhaustive_limit(4)
                      .build();
    solver.solve(*fixture.fluid, *fixture.universe, fixture.searcher->view(), 0, 1.0e-4f);
    const auto after_first = read_velocities(*fixture.fluid);
    const auto& counts = fixture.universe->state<UniverseCollisionCountState>()->data();
    const HostBuffer<int> first_counts(counts.begin(), counts.end());
    ASSERT_GT(first_counts[0], 64);
    EXPECT_EQ(first_counts[1], 73);
    EXPECT_TRUE(velocities_changed(before, after_first, 0, 3));
    EXPECT_FALSE(velocities_changed(before, after_first, 3, 3));

    solver.solve(*fixture.fluid, *fixture.universe, fixture.searcher->view(), 1, 1.0e-4f);
    const auto after_second = read_velocities(*fixture.fluid);
    const HostBuffer<int> second_counts(counts.begin(), counts.end());
    EXPECT_EQ(second_counts[0], first_counts[0]);
    ASSERT_GT(second_counts[1], 64);
    EXPECT_FALSE(velocities_changed(after_first, after_second, 0, 3));
    EXPECT_TRUE(velocities_changed(after_first, after_second, 3, 3));
}
