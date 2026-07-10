#include <atlas/universe/universe.h>

#include <atlas/geometry/geometry.h>
#include <atlas/logging/logging.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas {

Universe::Universe(const Float3& lower_corner,
                   const Float3& upper_corner,
                   const float cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

    _cell_volume = _cell_size * _cell_size * _cell_size;
    _inv_h       = 1.0f / _cell_size;
    _grid_size   = compute_grid_size(_lower_corner, _upper_corner, _inv_h);
    _cell_count  = _grid_size.x * _grid_size.y * _grid_size.z;
}

Universe::Builder
Universe::builder() noexcept {
    return Builder {};
}

Int3
Universe::compute_grid_size(const Float3& lower_corner,
                            const Float3& upper_corner,
                            const float inverse_cell_size) noexcept {
    return atlas::to_vector3i(atlas::floor((upper_corner - lower_corner) * inverse_cell_size))
        + Int3(1, 1, 1);
}

int
Universe::cell_count() const noexcept {
    return _cell_count;
}

Float3
Universe::lower_corner() const noexcept {
    return _lower_corner;
}

Float3
Universe::upper_corner() const noexcept {
    return _upper_corner;
}

Int3
Universe::grid_size() const noexcept {
    return _grid_size;
}

float
Universe::cell_size() const noexcept {
    return _cell_size;
}

float
Universe::cell_volume() const noexcept {
    return _cell_volume;
}

float
Universe::inverse_cell_size() const noexcept {
    return _inv_h;
}

UniverseStateStore&
Universe::states() noexcept {
    return _states;
}

const UniverseStateStore&
Universe::states() const noexcept {
    return _states;
}

Universe::Builder&
Universe::Builder::with_geometry(const Geometry& geometry) {
    const auto bound = geometry.bound();
    _lower_corner    = bound.lower_corner;
    _upper_corner    = bound.upper_corner;
    return *this;
}

Universe::Builder&
Universe::Builder::with_lower_corner(const Float3& v) noexcept {
    _lower_corner = v;
    return *this;
}

Universe::Builder&
Universe::Builder::with_upper_corner(const Float3& v) noexcept {
    _upper_corner = v;
    return *this;
}

Universe::Builder&
Universe::Builder::with_cell_size(const float h) noexcept {
    _cell_size = h;
    return *this;
}

Universe
Universe::Builder::build() const {
    validate();

    return Universe(_lower_corner, _upper_corner, _cell_size);
}

atlas::host_unique_ptr<Universe>
Universe::Builder::make_host_unique() const {
    return atlas::make_host_unique<Universe>(build());
}

void
Universe::Builder::validate() const {
    atlas::check<std::invalid_argument>(_cell_size > 0.0f)
        << "Universe::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(
        atlas::all(_upper_corner > _lower_corner))
        << "Universe::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const float inv_h = 1.0f / _cell_size;

    const Int3 gs = Universe::compute_grid_size(_lower_corner, _upper_corner, inv_h);

    atlas::check<std::invalid_argument>(atlas::all(gs >= Int3(1, 1, 1)))
        << "Universe::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Universe::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const long long cells64 = nx * ny * nz;

    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Universe::Builder validation failed: cell_count overflow/invalid. "
        << "cell_count=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

}
