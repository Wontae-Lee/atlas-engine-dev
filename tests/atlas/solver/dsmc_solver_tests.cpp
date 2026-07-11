#include <atlas/solver/dsmc/dsmc_solver.h>

#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver_type.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::DsmcKernelType;
using atlas::DsmcSolver;
using atlas::SolverType;

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
