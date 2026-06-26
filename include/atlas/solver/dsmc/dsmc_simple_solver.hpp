#pragma once

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
typename DsmcSimpleSolver<T>::Builder
DsmcSimpleSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
bool
DsmcSimpleSolver<T>::measure_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                                  const int index,
                                                  const T dt) {
    return _simple_statistics.measure(this->_probe, allocated_solver, index, dt);
}

template <typename T>
typename DsmcSimpleSolver<T>::Builder&
DsmcSimpleSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcSimpleSolver<T>::Builder&
DsmcSimpleSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcSimpleSolver<T>::Builder&
DsmcSimpleSolver<T>::Builder::with_searcher(SearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcSimpleSolver<T>::Builder&
DsmcSimpleSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename DsmcSimpleSolver<T>::Builder&
DsmcSimpleSolver<T>::Builder::with_workload_type(const DsmcCollisionWorkloadType workload_type) noexcept {
    _workload_type = workload_type;
    return *this;
}

template <typename T>
void
DsmcSimpleSolver<T>::Builder::validate() const {
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

template <typename T>
DsmcSimpleSolver<T>
DsmcSimpleSolver<T>::Builder::build() const {
    validate();
    return DsmcSimpleSolver<T>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcSimpleSolver<T>>
DsmcSimpleSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcSimpleSolver<T>>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

}