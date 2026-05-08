#pragma once
#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
namespace atlas::system {
template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(UniverseHostPtr<T> domain,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              T characteristic_length)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length) {
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";
    const auto num_of_cells = this->_universe->number_of_cells();
    if (!this->_universe->template has_state<atlas::universe::UniverseKnudsenNumberState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseKnudsenNumberState<T>>(
            static_cast<std::size_t>(num_of_cells));
    }
    const HostBuffer<T> kn_split { T(0.01), T(0.1), T(1.0) };
    d_kn_split = DeviceBuffer<T>(kn_split.begin(), kn_split.end());
}

template <typename T>
void
KnudsenCodec<T>::encode() {
    typename Codec<T>::CodecProbe probe;
    if (!this->make_probe(probe)
        || probe.temperature_ptr == nullptr
        || probe.number_particle_ptr == nullptr
        || probe.knudsen_number_ptr == nullptr) {
        return;
    }

    const T characteristic_length   = _characteristic_length;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1) {
                return;
            }
            const T number_density = probe.cell_volume > T(0)
                ? probe.number_particle_ptr[cell] * probe.statistical_weight / probe.cell_volume
                : T(0);
            if (!(number_density > T(0)) || !(probe.temperature_ptr[cell] > T(0)) || !(characteristic_length > T(0))) {
                probe.knudsen_number_ptr[cell] = T(0);
                return;
            }
            const T mean_free_path
                = static_cast<T>(atlas::boltzmann_constant) * probe.temperature_ptr[cell] / number_density;
            probe.knudsen_number_ptr[cell] = mean_free_path / characteristic_length;
        });
}

template <typename T>
void
KnudsenCodec<T>::decode() {
    typename Codec<T>::CodecProbe probe;
    if (!this->make_probe(probe)
        || probe.knudsen_number_ptr == nullptr
        || probe.allocated_solver_ptr == nullptr
        || d_kn_split.empty()) {
        return;
    }
    const auto* kn_split_ptr       = atlas::raw_pointer_cast(d_kn_split.data());
    const int split_count          = static_cast<int>(d_kn_split.size());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1) {
                if (probe.fixed_solver_ptr != nullptr) {
                    probe.allocated_solver_ptr[cell] = probe.fixed_solver_ptr[cell];
                }
                return;
            }
            const T kn           = probe.knudsen_number_ptr[cell];
            int allocated_solver = 0;
            while (allocated_solver < split_count && !(kn < kn_split_ptr[allocated_solver])) {
                ++allocated_solver;
            }
            probe.allocated_solver_ptr[cell] = allocated_solver;
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
KnudsenCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
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
    auto codec = KnudsenCodec<T>(_domain, _fluid, _searcher, _characteristic_length);
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
    auto codec = atlas::make_host_shared<KnudsenCodec<T>>(_domain, _fluid, _searcher, _characteristic_length);
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }
    return codec;
}

}
