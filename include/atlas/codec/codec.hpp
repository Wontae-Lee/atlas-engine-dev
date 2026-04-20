#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(UniverseHostPtr<T> domain,
                FluidHostPtr<T> fluid,
                SpatialHashingSearcherHostPtr<T> searcher)
    : _universe(std::move(domain))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {

    // Validate all required external dependencies immediately.
    //
    // A codec is not meaningful without:
    // - a universe that defines the simulation domain and cell layout
    // - a fluid that provides the simulation state to encode/decode
    // - a spatial hashing searcher that provides cell-based indexing support
    //
    // These checks ensure that the codec cannot be constructed in a partially
    // valid state that would later fail deep inside encode()/decode().
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "Codec: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "Codec: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "Codec: searcher must not be null.";

    // Initialize internal device-side buffers so the object starts in a
    // consistent state before any encode/decode operation is attempted.
    reset();
}

template <typename T>
void
Codec<T>::update() {

    // Perform one complete codec cycle.
    //
    // The intended high-level workflow is:
    // 1. encode the current simulation state into an intermediate form
    // 2. decode the processed/intermediate form back into simulation state
    //
    // This base implementation defines the canonical order.
    // Derived classes may override update() if they need additional steps,
    // custom synchronization, or a different execution schedule.
    this->encode();
    this->decode();
}

template <typename T>
void
Codec<T>::reset() noexcept {

    // Resize the per-cell solver allocation buffer to match the current
    // number of cells defined by the universe.
    //
    // The buffer is initialized with zeros so that all cells start in an
    // "unassigned" or "default allocation" state.
    //
    // The exact interpretation of each entry depends on the derived codec,
    // but a zero-initialized buffer provides a safe baseline.
    const auto num_of_cells = _universe->number_of_cells();

    d_allocated_solver.resize(num_of_cells, 0);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::allocated_solver() noexcept {

    // Return mutable access so derived systems or external host-side
    // orchestration code can update the solver allocation metadata directly.
    return d_allocated_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::allocated_solver() const noexcept {

    // Return read-only access for inspection without allowing modification.
    return d_allocated_solver;
}

}