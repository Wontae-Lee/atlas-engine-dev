#include <atlas/solver/solver.h>

#include <utility>

namespace atlas {

Solver::Solver(UniverseHostPtr universe, FluidHostPtr fluid, SearcherHostPtr searcher) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
}

void
Solver::solve(const float) {
}

void
Solver::solve(const DeviceBuffer<int>*, const int, const float) {
}

}
