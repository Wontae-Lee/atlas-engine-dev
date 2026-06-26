#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(UniverseHostPtr<T> domain,
                              FluidHostPtr<T> fluid,
                              SearcherHostPtr<T> searcher,
                              T characteristic_length,
                              T representative_collision_cross_sectional_area)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length)
    , _representative_collision_cross_sectional_area(representative_collision_cross_sectional_area) {
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";
    atlas::check<std::invalid_argument>(representative_collision_cross_sectional_area > T(0))
        << "KnudsenCodec: representative_collision_cross_sectional_area must be positive.";

    if (this->_universe->template state<UniverseKnudsenNumberState<T>>() == nullptr) {
        this->_universe->template emplace_state<UniverseKnudsenNumberState<T>>(
            static_cast<std::size_t>(this->_universe->number_of_cells()));
    }

    const HostBuffer<T> kn_split { T(0.01), T(0.1), T(1.0) };
    d_kn_split = DeviceBuffer<T>(kn_split.begin(), kn_split.end());
}

template <typename T>
void
KnudsenCodec<T>::encode() {
    if (!this->make_probe()
        || this->_probe.number_particle_ptr == nullptr
        || this->_probe.knudsen_number_ptr == nullptr) {
        return;
    }
    const auto probe = this->_probe;

    const T characteristic_length                         = _characteristic_length;
    const T representative_collision_cross_sectional_area = _representative_collision_cross_sectional_area;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (KnudsenCodec<T>::fixed_cell(probe, cell)) {
                return;
            }

            probe.knudsen_number_ptr[cell] = KnudsenCodec<T>::knudsen_number(
                probe.number_particle_ptr[cell],
                probe.statistical_weight,
                probe.cell_volume,
                characteristic_length,
                representative_collision_cross_sectional_area);
        });
}

template <typename T>
void
KnudsenCodec<T>::decode() {
    if (!this->make_probe()
        || this->_probe.knudsen_number_ptr == nullptr
        || this->_probe.allocated_solver_ptr == nullptr
        || d_kn_split.empty()) {
        return;
    }
    const auto probe = this->_probe;

    const auto* kn_split_ptr = atlas::raw_pointer_cast(d_kn_split.data());
    const int split_count    = static_cast<int>(d_kn_split.size());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (KnudsenCodec<T>::fixed_cell(probe, cell)) {
                if (probe.fixed_solver_ptr != nullptr) {
                    probe.allocated_solver_ptr[cell] = probe.fixed_solver_ptr[cell];
                }
                return;
            }

            probe.allocated_solver_ptr[cell] = KnudsenCodec<T>::solver_index(
                probe.knudsen_number_ptr[cell],
                kn_split_ptr,
                split_count);
        });
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_searcher(SearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {
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
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    _fixed_region = std::move(fixed_region);
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KnudsenCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "KnudsenCodec::Builder: searcher must not be null.";
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
    atlas::check<std::invalid_argument>(_representative_collision_cross_sectional_area > T(0))
        << "KnudsenCodec::Builder: representative_collision_cross_sectional_area must be positive.";

    const auto cell_count = static_cast<std::size_t>(_domain->number_of_cells());
    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_solver size must match universe cell count.";
    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_region size must match universe cell count.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {
    validate();

    auto codec = KnudsenCodec<T>(
        _domain,
        _fluid,
        _searcher,
        _characteristic_length,
        _representative_collision_cross_sectional_area);
    if (!_fixed_solver.empty()) {
        codec.set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec.set_fixed_region(_fixed_region);
    }

    return codec;
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {
    validate();

    auto codec = atlas::make_host_shared<KnudsenCodec<T>>(
        _domain,
        _fluid,
        _searcher,
        _characteristic_length,
        _representative_collision_cross_sectional_area);
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }

    return codec;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
KnudsenCodec<T>::fixed_cell(const CodecProbe<T>& probe, const int cell) noexcept {
    return probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE
    T
    KnudsenCodec<T>::knudsen_number(const T particle_count,
                                    const T statistical_weight,
                                    const T cell_volume,
                                    const T characteristic_length,
                                    const T representative_collision_cross_sectional_area) noexcept {
    const T number_density = cell_volume > T(0)
        ? particle_count * statistical_weight / cell_volume
        : T(0);

    if (!(number_density > T(0))
        || !(characteristic_length > T(0))
        || !(representative_collision_cross_sectional_area > T(0))) {
        return T(0);
    }

    const T mean_free_path = T(1)
        / (static_cast<T>(atlas::SQRT_TWO)
           * number_density
           * representative_collision_cross_sectional_area);
    return mean_free_path / characteristic_length;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
KnudsenCodec<T>::solver_index(const T kn, const T* splits, const int split_count) noexcept {
    int index = 0;
    while (index < split_count && !(kn < splits[index])) {
        ++index;
    }
    return index;
}

}