#include "../../utilities/test_utils.h"

#include <atlas/solver/dsmc/dsmc_energy_exchange_solver.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_simple_solver.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using DsmcSolver        = atlas::system::DsmcSolver<float>;
using DsmcSimpleSolver  = atlas::system::DsmcSimpleSolver<float>;
using DsmcEnergyExchangeSolver = atlas::system::DsmcEnergyExchangeSolver<float>;
using DsmcFlattenWorkload = atlas::system::DsmcFlattenWorkload<float>;
using DsmcStatistics = atlas::system::DsmcStatistics<float>;
using DsmcSimpleStatistics = atlas::system::DsmcSimpleStatistics<float>;

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
