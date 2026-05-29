#include "../../utilities/test_utils.h"

#include <atlas/solver/dsmc/dsmc_flatten_solver.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using DsmcSolver        = atlas::system::DsmcSolver<float>;
using DsmcFlattenSolver = atlas::system::DsmcFlattenSolver<float>;

static_assert(std::is_base_of_v<DsmcSolver, DsmcFlattenSolver>);

}

TEST(DsmcFlattenSolver, InheritsDsmcSolverCollisionExecution) {
    EXPECT_TRUE((std::is_base_of_v<DsmcSolver, DsmcFlattenSolver>));
}
