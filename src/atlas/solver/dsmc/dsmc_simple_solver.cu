#include <atlas/solver/dsmc/dsmc_simple_solver.h>

#include <stdexcept>
#include <utility>

namespace atlas {

DsmcSimpleSolver::Builder
DsmcSimpleSolver::builder() noexcept {
    return Builder {};
}

bool
DsmcSimpleSolver::measure_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                               const int index,
                                               const float dt) {
    return _simple_statistics.measure(this->_probe, allocated_solver, index, dt);
}

DsmcSimpleSolver::Builder&
DsmcSimpleSolver::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

DsmcSimpleSolver::Builder&
DsmcSimpleSolver::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

DsmcSimpleSolver::Builder&
DsmcSimpleSolver::Builder::with_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

DsmcSimpleSolver::Builder&
DsmcSimpleSolver::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

DsmcSimpleSolver::Builder&
DsmcSimpleSolver::Builder::with_workload_type(const DsmcCollisionWorkloadType workload_type) noexcept {
    _workload_type = workload_type;
    return *this;
}

void
DsmcSimpleSolver::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcSimpleSolver::Builder: universe must not be null.");
    }
    if (!_fluid) {
        throw std::runtime_error("DsmcSimpleSolver::Builder: fluid must not be null.");
    }
    if (!_searcher) {
        throw std::runtime_error("DsmcSimpleSolver::Builder: searcher must not be null.");
    }
}

DsmcSimpleSolver
DsmcSimpleSolver::Builder::build() const {
    validate();
    return DsmcSimpleSolver(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

atlas::host_shared_ptr<DsmcSimpleSolver>
DsmcSimpleSolver::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcSimpleSolver>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

}
