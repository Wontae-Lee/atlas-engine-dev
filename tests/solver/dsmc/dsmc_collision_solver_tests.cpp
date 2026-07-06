#include <atlas/solver/dsmc/dsmc_solver.h>

#include <atlas/solver/dsmc/dsmc_energy_exchange_solver.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_simple_solver.h>
#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::DsmcEnergyExchangeSolver;
using atlas::DsmcFlattenWorkload;
using atlas::DsmcSimpleSolver;
using atlas::DsmcSimpleStatistics;
using atlas::DsmcSolver;
using atlas::DsmcStatistics;

static_assert(std::is_base_of_v<DsmcSolver, DsmcSimpleSolver>);
static_assert(std::is_base_of_v<DsmcSolver, DsmcEnergyExchangeSolver>);
static_assert(!std::is_base_of_v<DsmcSolver, DsmcFlattenWorkload>);
static_assert(std::is_base_of_v<DsmcStatistics, DsmcSimpleStatistics>);

}

TEST(DsmcSimpleSolver, InheritsDsmcSolverCollisionExecution) {
    EXPECT_TRUE((std::is_base_of_v<DsmcSolver, DsmcSimpleSolver>));
}

TEST(DsmcEnergyExchangeSolver, InheritsDsmcSolverCollisionExecution) {
    EXPECT_TRUE((std::is_base_of_v<DsmcSolver, DsmcEnergyExchangeSolver>));
}

TEST(DsmcFlattenWorkload, IsCollisionWorkloadOption) {
    EXPECT_FALSE((std::is_base_of_v<DsmcSolver, DsmcFlattenWorkload>));
}

TEST(DsmcSimpleStatistics, InheritsDsmcStatistics) {
    EXPECT_TRUE((std::is_base_of_v<DsmcStatistics, DsmcSimpleStatistics>));
}
