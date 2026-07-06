#include <atlas/solver/dsmc/dsmc_simple_solver.h>

#include "dsmc_test_utils.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(DsmcSimpleSolver, BuilderRejectsMissingDependencies) {
    EXPECT_THROW(atlas::DsmcSimpleSolver::builder().validate(), std::runtime_error);
}

TEST(DsmcSimpleSolver, BuilderCreatesConfiguredSolver) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);

    const auto solver = atlas::DsmcSimpleSolver::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_searcher(searcher)
        .with_kernel_type(atlas::DsmcKernelType::variable_soft_sphere)
        .with_workload_type(atlas::DsmcCollisionWorkloadType::flatten)
        .build();

    EXPECT_EQ(solver.kernel_type(), atlas::DsmcKernelType::variable_soft_sphere);
    EXPECT_EQ(solver.workload_type(), atlas::DsmcCollisionWorkloadType::flatten);
}

TEST(DsmcSimpleSolver, BuilderCreatesHostSharedSolver) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);

    const auto solver = atlas::DsmcSimpleSolver::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_searcher(searcher)
        .make_host_shared();

    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->kernel_type(), atlas::DsmcKernelType::hard_sphere);
}

TEST(DsmcSimpleSolver, MeasureCollisionStatisticsUsesSimpleStatisticComponent) {
    atlas::DsmcSimpleSolver solver;

    EXPECT_TRUE(solver.measure_collision_statistics(nullptr, 0, 0.0f));
}
