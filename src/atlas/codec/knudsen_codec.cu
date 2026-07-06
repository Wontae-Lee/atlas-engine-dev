#include <atlas/codec/knudsen_codec.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

KnudsenCodec::Builder
KnudsenCodec::builder() noexcept {
    return Builder {};
}

KnudsenCodec::KnudsenCodec(UniverseHostPtr domain,
                           FluidHostPtr fluid,
                           SearcherHostPtr searcher,
                           const float characteristic_length,
                           const float representative_collision_cross_sectional_area)
    : Codec(std::move(domain), std::move(fluid), std::move(searcher))
    , _characteristic_length(characteristic_length)
    , _representative_collision_cross_sectional_area(representative_collision_cross_sectional_area) {
    atlas::check<std::invalid_argument>(characteristic_length > 0.0f)
        << "KnudsenCodec: characteristic_length must be positive.";
    atlas::check<std::invalid_argument>(representative_collision_cross_sectional_area > 0.0f)
        << "KnudsenCodec: representative_collision_cross_sectional_area must be positive.";

    if (_universe->state<UniverseKnudsenNumberState>() == nullptr) {
        _universe->emplace_state<UniverseKnudsenNumberState>(
            static_cast<std::size_t>(_universe->cell_count()));
    }

    const HostBuffer<float> kn_split { 0.01f, 0.1f, 1.0f };
    d_kn_split = DeviceBuffer<float>(kn_split.begin(), kn_split.end());
}

void
KnudsenCodec::encode() {
    if (!make_probe()
        || _probe.number_particle_ptr == nullptr
        || _probe.knudsen_number_ptr == nullptr) {
        return;
    }

    encode_cells();
}

void
KnudsenCodec::encode_cells() {
    const auto probe = _probe;

    const float characteristic_length                         = _characteristic_length;
    const float representative_collision_cross_sectional_area = _representative_collision_cross_sectional_area;

    // Cells flagged fixed_region are exempt from the physical Knudsen
    // number computation entirely — their allocated_solver is pinned by
    // decode_cells() directly from fixed_solver_ptr instead.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            if (KnudsenCodec::fixed_cell(probe, cell)) {
                return;
            }

            probe.knudsen_number_ptr[cell] = KnudsenCodec::knudsen_number(
                probe.number_particle_ptr[cell],
                probe.statistical_weight,
                probe.cell_volume,
                characteristic_length,
                representative_collision_cross_sectional_area);
        });
}

void
KnudsenCodec::decode() {
    if (!make_probe()
        || _probe.knudsen_number_ptr == nullptr
        || _probe.allocated_solver_ptr == nullptr
        || d_kn_split.empty()) {
        return;
    }

    decode_cells();
}

void
KnudsenCodec::decode_cells() {
    const auto probe = _probe;

    const auto* kn_split_ptr = atlas::raw_pointer_cast(d_kn_split.data());
    const int split_count    = static_cast<int>(d_kn_split.size());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            if (KnudsenCodec::fixed_cell(probe, cell)) {
                if (probe.fixed_solver_ptr != nullptr) {
                    probe.allocated_solver_ptr[cell] = probe.fixed_solver_ptr[cell];
                }
                return;
            }

            probe.allocated_solver_ptr[cell] = KnudsenCodec::solver_index(
                probe.knudsen_number_ptr[cell],
                kn_split_ptr,
                split_count);
        });
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_domain(UniverseHostPtr domain) noexcept {
    _domain = std::move(domain);
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_characteristic_length(const float characteristic_length) noexcept {
    _characteristic_length = characteristic_length;
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_representative_collision_cross_sectional_area(
    const float representative_collision_cross_sectional_area) noexcept {
    _representative_collision_cross_sectional_area = representative_collision_cross_sectional_area;
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept {
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

KnudsenCodec::Builder&
KnudsenCodec::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    _fixed_region = std::move(fixed_region);
    return *this;
}

void
KnudsenCodec::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KnudsenCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "KnudsenCodec::Builder: searcher must not be null.";
    atlas::check<std::invalid_argument>(_characteristic_length > 0.0f)
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
    atlas::check<std::invalid_argument>(_representative_collision_cross_sectional_area > 0.0f)
        << "KnudsenCodec::Builder: representative_collision_cross_sectional_area must be positive.";

    const auto cell_count = static_cast<std::size_t>(_domain->cell_count());
    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_solver size must match universe cell count.";
    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "KnudsenCodec::Builder: fixed_region size must match universe cell count.";
}

KnudsenCodec
KnudsenCodec::Builder::build() const {
    validate();

    auto codec = KnudsenCodec(
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

atlas::host_shared_ptr<KnudsenCodec>
KnudsenCodec::Builder::make_host_shared() const {
    validate();

    auto codec = atlas::make_host_shared<KnudsenCodec>(
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

}
