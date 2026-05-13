#pragma once
#include <atlas/memory/raw_pointer_cast.h>
namespace atlas::system {
template <typename T>
Codec<T>::Codec(UniverseHostPtr<T> domain,
                FluidHostPtr<T> fluid,
                SpatialHashingSearcherHostPtr<T> searcher)
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
    this->encode();
    this->decode();
}

template <typename T>
void
Codec<T>::reset() noexcept {
    const auto num_of_cells = _universe->number_of_cells();
    d_allocated_solver.resize(num_of_cells, 0);
    d_fixed_solver.resize(num_of_cells, 0);
    d_fixed_region.resize(num_of_cells, 0);
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
Codec<T>::make_probe(CodecProbe& probe) noexcept {
    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* temperature_state
        = _universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* number_particle_state
        = _universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* knudsen_number_state
        = _universe->template state<atlas::universe::UniverseKnudsenNumberState<T>>();

    probe.temperature_ptr      = temperature_state != nullptr
             ? atlas::raw_pointer_cast(temperature_state->data().data())
             : nullptr;
    probe.number_particle_ptr  = number_particle_state != nullptr
         ? atlas::raw_pointer_cast(number_particle_state->data().data())
         : nullptr;
    probe.knudsen_number_ptr   = knudsen_number_state != nullptr
          ? atlas::raw_pointer_cast(knudsen_number_state->data().data())
          : nullptr;
    probe.allocated_solver_ptr = d_allocated_solver.empty()
        ? nullptr
        : atlas::raw_pointer_cast(d_allocated_solver.data());
    probe.fixed_solver_ptr     = d_fixed_solver.empty()
            ? nullptr
            : atlas::raw_pointer_cast(d_fixed_solver.data());
    probe.fixed_region_ptr     = d_fixed_region.empty()
            ? nullptr
            : atlas::raw_pointer_cast(d_fixed_region.data());
    probe.indices_ptr          = _searcher->indices();
    probe.cell_start_ptr       = _searcher->cell_start();
    probe.cell_end_ptr         = _searcher->cell_end();
    probe.particle_count       = static_cast<int>(_fluid->particle_count());
    probe.num_of_cells         = _universe->number_of_cells();
    probe.cell_volume          = _universe->cell_volume();
    probe.statistical_weight   = _fluid->statistical_weight();

    return probe.num_of_cells > 0;
}

}
