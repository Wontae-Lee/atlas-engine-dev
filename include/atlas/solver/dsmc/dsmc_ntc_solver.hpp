#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/scan/exclusive_scan.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcNtcSolver<T>::DsmcNtcSolver(atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
                                DsmcKernel<T> op)
    : DsmcSolver<T>(std::move(fluid), std::move(op)) { }

template <typename T>
typename DsmcNtcSolver<T>::Builder
DsmcNtcSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcNtcSolver<T>::solve(Universe<T>& domain,
                        SpatialHashingProbe<T>& searcher,
                        FluidDeviceProbe<T>& particle,
                        CodecDeviceProbe<T>&) {
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_operator(const DsmcKernel<T>& op) noexcept {
    _operator = op;
    return *this;
}

template <typename T>
void
DsmcNtcSolver<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("DsmcNtcSolver::Builder: fluid must not be null.");
    }
}

template <typename T>
DsmcNtcSolver<T>
DsmcNtcSolver<T>::Builder::build() const {
    validate();
    return DsmcNtcSolver<T>(_fluid, _operator);
}

template <typename T>
atlas::host_shared_ptr<DsmcNtcSolver<T>>
DsmcNtcSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcNtcSolver<T>>(_fluid, _operator);
}

}