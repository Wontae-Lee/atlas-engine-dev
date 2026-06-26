#pragma once

#include <atlas/codec/detail/codec_probe_builder.h>
#include <atlas/logging/logging.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
Codec<T>::Codec(UniverseHostPtr<T> domain,
                FluidHostPtr<T> fluid,
                SearcherHostPtr<T> searcher)
    : _universe(std::move(domain))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "Codec: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "Codec: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "Codec: searcher must not be null.";

    reset();
}

template <typename T>
void
Codec<T>::update() {
    encode();
    decode();
}

template <typename T>
void
Codec<T>::reset() noexcept {
    const auto cell_count = _universe->number_of_cells();
    d_allocated_solver.resize(cell_count, 0);
    d_fixed_solver.resize(cell_count, 0);
    d_fixed_region.resize(cell_count, 0);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::allocated_solver() noexcept {
    return d_allocated_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::allocated_solver() const noexcept {
    return d_allocated_solver;
}

template <typename T>
void
Codec<T>::set_fixed_solver(DeviceBuffer<int> fixed_solver) {
    if (_universe && !fixed_solver.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_solver.size() == static_cast<std::size_t>(_universe->number_of_cells()))
            << "Codec: fixed_solver size must match universe cell count.";
    }
    d_fixed_solver = std::move(fixed_solver);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::fixed_solver() noexcept {
    return d_fixed_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::fixed_solver() const noexcept {
    return d_fixed_solver;
}

template <typename T>
void
Codec<T>::set_fixed_region(DeviceBuffer<int> fixed_region) {
    if (_universe && !fixed_region.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_region.size() == static_cast<std::size_t>(_universe->number_of_cells()))
            << "Codec: fixed_region size must match universe cell count.";
    }
    d_fixed_region = std::move(fixed_region);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::fixed_region() noexcept {
    return d_fixed_region;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::fixed_region() const noexcept {
    return d_fixed_region;
}

template <typename T>
bool
Codec<T>::make_probe() noexcept {
    return detail::CodecProbeBuilder<T>::make(
        _probe,
        _universe,
        _fluid,
        _searcher,
        d_allocated_solver,
        d_fixed_solver,
        d_fixed_region);
}

}