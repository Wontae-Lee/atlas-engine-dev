#pragma once

namespace atlas {

// Indexes the solvers an orchestrator drives. A cell's UniverseAllocatedSolverState
// entry is an index into that list, so the order the solvers are registered in
// is what the codec's bucketing refers to.
enum class SolverType : int {

    dsmc
};

}
