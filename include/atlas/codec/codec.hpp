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
    // A codec requires all simulation-side resources because encoding and
    // decoding decisions are evaluated from universe, fluid, and searcher data.
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "Codec: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "Codec: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "Codec: searcher must not be null.";

    // Allocate solver-control buffers to match the current universe layout.
    reset();
}

template <typename T>
void
Codec<T>::update() {
    // Recompute solver allocation first, then apply the decoded solver state
    // back to the simulation data structures.
    this->encode();
    this->decode();
}

template <typename T>
void
Codec<T>::reset() noexcept {
    const auto num_of_cells = _universe->number_of_cells();

    // Each cell stores one solver-selection or constraint flag.
    d_allocated_solver.resize(num_of_cells, 0);
    d_fixed_solver.resize(num_of_cells, 0);
    d_fixed_region.resize(num_of_cells, 0);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::allocated_solver() noexcept {
    // Expose mutable solver assignments for derived codecs or host-side setup.
    return d_allocated_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::allocated_solver() const noexcept {
    // Expose read-only solver assignments for diagnostics and external queries.
    return d_allocated_solver;
}

template <typename T>
void
Codec<T>::set_fixed_solver(DeviceBuffer<int> fixed_solver) {
    // A non-empty fixed-solver map must define exactly one entry per cell.
    if (_universe && !fixed_solver.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_solver.size() == static_cast<std::size_t>(_universe->number_of_cells()))
            << "Codec: fixed_solver size must match universe cell count.";
    }

    // Replace the current fixed-solver constraints without copying the buffer.
    d_fixed_solver = std::move(fixed_solver);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::fixed_solver() noexcept {
    // Expose mutable fixed-solver constraints for explicit user configuration.
    return d_fixed_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::fixed_solver() const noexcept {
    // Expose read-only fixed-solver constraints for inspection.
    return d_fixed_solver;
}

template <typename T>
void
Codec<T>::set_fixed_region(DeviceBuffer<int> fixed_region) {
    // A non-empty fixed-region map must define exactly one entry per cell.
    if (_universe && !fixed_region.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_region.size() == static_cast<std::size_t>(_universe->number_of_cells()))
            << "Codec: fixed_region size must match universe cell count.";
    }

    // Replace the current fixed-region constraints without copying the buffer.
    d_fixed_region = std::move(fixed_region);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::fixed_region() noexcept {
    // Expose mutable fixed-region constraints for explicit user configuration.
    return d_fixed_region;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::fixed_region() const noexcept {
    // Expose read-only fixed-region constraints for inspection.
    return d_fixed_region;
}

template <typename T>
bool
Codec<T>::make_probe() noexcept {
    // Rebuild the probe from scratch so stale pointers are never reused.
    _probe = {};

    // A valid probe cannot be formed unless all required simulation resources exist.
    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    // Retrieve optional universe states used by codec kernels.
    auto* temperature_state
        = _universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* number_particle_state
        = _universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* knudsen_number_state
        = _universe->template state<atlas::universe::UniverseKnudsenNumberState<T>>();

    // Store raw device pointers when the corresponding state is available.
    _probe.temperature_ptr = temperature_state != nullptr
        ? atlas::raw_pointer_cast(temperature_state->data().data())
        : nullptr;

    _probe.number_particle_ptr = number_particle_state != nullptr
        ? atlas::raw_pointer_cast(number_particle_state->data().data())
        : nullptr;

    _probe.knudsen_number_ptr = knudsen_number_state != nullptr
        ? atlas::raw_pointer_cast(knudsen_number_state->data().data())
        : nullptr;

    // Empty buffers are represented as null pointers to keep kernel-side checks simple.
    _probe.allocated_solver_ptr = d_allocated_solver.empty()
        ? nullptr
        : atlas::raw_pointer_cast(d_allocated_solver.data());

    _probe.fixed_solver_ptr = d_fixed_solver.empty()
        ? nullptr
        : atlas::raw_pointer_cast(d_fixed_solver.data());

    _probe.fixed_region_ptr = d_fixed_region.empty()
        ? nullptr
        : atlas::raw_pointer_cast(d_fixed_region.data());

    // Attach spatial-hashing views required for cell-wise particle traversal.
    _probe.indices_ptr    = _searcher->indices();
    _probe.cell_start_ptr = _searcher->cell_start();
    _probe.cell_end_ptr   = _searcher->cell_end();

    // Cache scalar metadata so kernels can access simulation constants without
    // dereferencing host-side objects.
    _probe.particle_count     = static_cast<int>(_fluid->particle_count());
    _probe.num_of_cells       = _universe->number_of_cells();
    _probe.cell_volume        = _universe->cell_volume();
    _probe.statistical_weight = _fluid->statistical_weight();

    // A probe with zero cells cannot produce meaningful codec work.
    return _probe.num_of_cells > 0;
}

} // namespace atlas::system