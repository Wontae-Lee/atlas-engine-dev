#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

namespace atlas {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the codec through a fluent API.
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(UniverseHostPtr<T> domain,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              T characteristic_length,
                              T representative_collision_cross_sectional_area)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length)
    , _representative_collision_cross_sectional_area(representative_collision_cross_sectional_area) {
    // The Knudsen number is defined as mean_free_path / characteristic_length,
    // so the reference length must be strictly positive.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    atlas::check<std::invalid_argument>(representative_collision_cross_sectional_area > T(0))
        << "KnudsenCodec: representative_collision_cross_sectional_area must be positive.";

    // Ensure that the universe owns a per-cell Knudsen number state.
    // The encoder writes computed Kn values into this state.
    const auto num_of_cells = this->_universe->number_of_cells();
    if (!this->_universe->template has_state<atlas::UniverseKnudsenNumberState<T>>()) {
        this->_universe->template emplace_state<atlas::UniverseKnudsenNumberState<T>>(
            static_cast<std::size_t>(num_of_cells));
    }

    // Define Knudsen-number thresholds used during decoding.
    // These split points map each cell to a solver index:
    //   Kn < 0.01  -> solver 0
    //   Kn < 0.1   -> solver 1
    //   Kn < 1.0   -> solver 2
    //   Kn >= 1.0  -> solver 3
    const HostBuffer<T> kn_split { T(0.01), T(0.1), T(1.0) };

    // Copy the thresholds to device memory so the decode kernel can access them.
    d_kn_split = DeviceBuffer<T>(kn_split.begin(), kn_split.end());
}

template <typename T>
void
KnudsenCodec<T>::encode() {
    // Encoding requires particle counts and writable Knudsen-number storage.
    // The current representative hard-sphere-style model does not use temperature explicitly.
    if (!this->make_probe()
        || this->_probe.number_particle_ptr == nullptr
        || this->_probe.knudsen_number_ptr == nullptr) {
        return;
    }
    const auto probe = this->_probe;

    // Capture model parameters by value so they are available inside the device kernel.
    const T characteristic_length                         = _characteristic_length;
    const T representative_collision_cross_sectional_area = _representative_collision_cross_sectional_area;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Fixed cells keep their previously encoded Knudsen value.
            if (probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1) {
                return;
            }

            // Convert simulation particles into a physical number density.
            //
            // number_density = particle_count * statistical_weight / cell_volume
            //
            // With SI-consistent inputs, number_density has units of 1 / m^3.
            const T number_density = probe.cell_volume > T(0)
                ? probe.number_particle_ptr[cell] * probe.statistical_weight / probe.cell_volume
                : T(0);

            // Invalid density or model scales produce a safe zero Knudsen value.
            if (!(number_density > T(0))
                || !(characteristic_length > T(0))
                || !(representative_collision_cross_sectional_area > T(0))) {
                probe.knudsen_number_ptr[cell] = T(0);
                return;
            }

            // Estimate a representative mean free path:
            //
            //   lambda = 1 / (sqrt(2) * n * sigma)
            //
            // where:
            //   n     is the number density,
            //   sigma is the representative collision cross-sectional area.
            //
            // This is a single representative hard-sphere-style estimate. For
            // multi-species gases, sigma should eventually be replaced by a
            // species-pair-dependent or temperature-dependent collision model.
            const T mean_free_path = T(1)
                / (static_cast<T>(atlas::SQRT_TWO)
                   * number_density
                   * representative_collision_cross_sectional_area);

            // Store the dimensionless Knudsen number:
            //
            //   Kn = lambda / L
            //
            // where L is the configured characteristic length.
            probe.knudsen_number_ptr[cell] = mean_free_path / characteristic_length;
        });
}

template <typename T>
void
KnudsenCodec<T>::decode() {
    // Decoding requires the computed Knudsen number, the solver-allocation state,
    // and the device-side split thresholds.
    if (!this->make_probe()
        || this->_probe.knudsen_number_ptr == nullptr
        || this->_probe.allocated_solver_ptr == nullptr
        || d_kn_split.empty()) {
        return;
    }
    const auto probe = this->_probe;

    // Expose the Kn split thresholds to the device kernel.
    const auto* kn_split_ptr = atlas::raw_pointer_cast(d_kn_split.data());
    const int split_count    = static_cast<int>(d_kn_split.size());

    // Assign one solver index per cell according to the local Knudsen number.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Fixed regions bypass automatic solver selection.
            // If a fixed solver field exists, use that solver index directly.
            if (probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1) {
                if (probe.fixed_solver_ptr != nullptr) {
                    probe.allocated_solver_ptr[cell] = probe.fixed_solver_ptr[cell];
                }
                return;
            }

            // Find the first threshold that is larger than the current Kn value.
            // The resulting index is used as the solver allocation.
            const T kn = probe.knudsen_number_ptr[cell];

            if (split_count == 3) {
                int allocated_solver = 3;
                if (kn < kn_split_ptr[0]) {
                    allocated_solver = 0;
                } else if (kn < kn_split_ptr[1]) {
                    allocated_solver = 1;
                } else if (kn < kn_split_ptr[2]) {
                    allocated_solver = 2;
                }
                probe.allocated_solver_ptr[cell] = allocated_solver;
                return;
            }

            int allocated_solver = 0;
            while (allocated_solver < split_count && !(kn < kn_split_ptr[allocated_solver])) {
                ++allocated_solver;
            }

            // Store the selected solver index for this cell.
            probe.allocated_solver_ptr[cell] = allocated_solver;
        });
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {
    // Store the universe that provides the cell layout and per-cell states.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid that provides particle-related states used by the codec.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial searcher required by the base Codec interface.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {
    // Store the reference length used to normalize the mean free path into a Knudsen number.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_representative_collision_cross_sectional_area(
    T representative_collision_cross_sectional_area) noexcept {
    _representative_collision_cross_sectional_area = representative_collision_cross_sectional_area;
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept {
    // Store optional per-cell solver indices for regions that should not be automatically decoded.
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    // Store optional per-cell flags identifying cells with manually fixed solver allocation.
    _fixed_region = std::move(fixed_region);
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {
    // A valid universe is required because Knudsen numbers and solver allocations are cell-based.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: universe must not be null.";

    // A valid fluid is required because the encoder reads temperature and particle-count states.
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KnudsenCodec::Builder: fluid must not be null.";

    // A valid searcher is required by the shared codec infrastructure.
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "KnudsenCodec::Builder: searcher must not be null.";

    // The reference length must be positive for the Knudsen-number normalization.
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";

    atlas::check<std::invalid_argument>(_representative_collision_cross_sectional_area > T(0))
        << "KnudsenCodec::Builder: representative_collision_cross_sectional_area must be positive.";

    // Optional fixed-region arrays are per-cell fields, so their size must match the universe cell count.
    const auto cell_count = static_cast<std::size_t>(_domain->number_of_cells());

    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_solver size must match universe cell count.";

    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_region size must match universe cell count.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {
    // Validate all required dependencies and optional per-cell fields before construction.
    validate();

    // Construct the codec with the required domain, fluid, searcher, and characteristic length.
    auto codec = KnudsenCodec<T>(
        _domain,
        _fluid,
        _searcher,
        _characteristic_length,
        _representative_collision_cross_sectional_area);

    // Attach optional fixed solver assignments after the base codec is created.
    if (!_fixed_solver.empty()) {
        codec.set_fixed_solver(_fixed_solver);
    }

    // Attach optional fixed-region flags after the base codec is created.
    if (!_fixed_region.empty()) {
        codec.set_fixed_region(_fixed_region);
    }

    return codec;
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {
    // Validate first so construction through shared ownership follows the same rules as build().
    validate();

    // Allocate the configured codec in host shared memory.
    auto codec = atlas::make_host_shared<KnudsenCodec<T>>(
        _domain,
        _fluid,
        _searcher,
        _characteristic_length,
        _representative_collision_cross_sectional_area);

    // Attach optional fixed solver assignments after construction.
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }

    // Attach optional fixed-region flags after construction.
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }

    return codec;
}

} // namespace atlas
