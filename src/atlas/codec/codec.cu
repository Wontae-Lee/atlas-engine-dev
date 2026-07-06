#include <atlas/codec/codec.h>

#include <atlas/codec/detail/codec_probe_builder.h>
#include <atlas/logging/logging.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

Codec::Codec(UniverseHostPtr domain,
             FluidHostPtr fluid,
             SearcherHostPtr searcher)
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

void
Codec::update() {
    encode();
    decode();
}

void
Codec::reset() noexcept {
    const auto cell_count = _universe->cell_count();
    d_allocated_solver.resize(cell_count, 0);
    d_fixed_solver.resize(cell_count, 0);
    d_fixed_region.resize(cell_count, 0);
}

DeviceBuffer<int>&
Codec::allocated_solver() noexcept {
    return d_allocated_solver;
}

const DeviceBuffer<int>&
Codec::allocated_solver() const noexcept {
    return d_allocated_solver;
}

void
Codec::set_fixed_solver(DeviceBuffer<int> fixed_solver) {
    if (_universe && !fixed_solver.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_solver.size() == static_cast<std::size_t>(_universe->cell_count()))
            << "Codec: fixed_solver size must match universe cell count.";
    }
    d_fixed_solver = std::move(fixed_solver);
}

DeviceBuffer<int>&
Codec::fixed_solver() noexcept {
    return d_fixed_solver;
}

const DeviceBuffer<int>&
Codec::fixed_solver() const noexcept {
    return d_fixed_solver;
}

void
Codec::set_fixed_region(DeviceBuffer<int> fixed_region) {
    if (_universe && !fixed_region.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_region.size() == static_cast<std::size_t>(_universe->cell_count()))
            << "Codec: fixed_region size must match universe cell count.";
    }
    d_fixed_region = std::move(fixed_region);
}

DeviceBuffer<int>&
Codec::fixed_region() noexcept {
    return d_fixed_region;
}

const DeviceBuffer<int>&
Codec::fixed_region() const noexcept {
    return d_fixed_region;
}

bool
Codec::make_probe() noexcept {
    return detail::CodecProbeBuilder::make(
        _probe,
        _universe,
        _fluid,
        _searcher,
        d_allocated_solver,
        d_fixed_solver,
        d_fixed_region);
}

}
