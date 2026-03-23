#pragma once
#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
Domain<T>::Domain(const Vector3<T>& lower_corner,
                  const Vector3<T>& upper_corner,
                  T cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    _cell_volume = _cell_size * _cell_size * _cell_size;
    _inv_h       = T(1) / _cell_size;

    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;

    d_temperature.resize(_num_of_cells, T(0));
    d_field_force.resize(_num_of_cells, Vector3<T> { T(0), T(0), T(0) });
}

template <typename T>
typename Domain<T>::Builder
Domain<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
DomainDeviceProbe<T>
Domain<T>::make_device_probe() noexcept {

    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    DomainDeviceProbe<T> probe;

    probe.temperature = atlas::raw_pointer_cast(d_temperature.data());
    probe.field_force = atlas::raw_pointer_cast(d_field_force.data());

    probe.lower_corner = _lower_corner;
    probe.upper_corner = _upper_corner;
    probe.grid_size    = _grid_size;

    probe.cell_size   = _cell_size;
    probe.cell_volume = _cell_volume;
    probe.inv_h       = _inv_h;

    probe.num_of_cells = _num_of_cells;

    return probe;
}

template <typename T>
int
Domain<T>::number_of_cells() const noexcept {

    return _num_of_cells;
}

template <typename T>
Vector3<T>
Domain<T>::lower_corner() const noexcept {

    return _lower_corner;
}

template <typename T>
Vector3<T>
Domain<T>::upper_corner() const noexcept {

    return _upper_corner;
}

template <typename T>
Vector3<int>
Domain<T>::grid_size() const noexcept {

    return _grid_size;
}

template <typename T>
T
Domain<T>::cell_size() const noexcept {

    return _cell_size;
}

template <typename T>
T
Domain<T>::cell_volume() const noexcept {

    return _cell_volume;
}

template <typename T>
T
Domain<T>::inverse_cell_size() const noexcept {

    return _inv_h;
}

template <typename T>
Domain<T>
Domain<T>::Builder::build() const {

    validate();
    return Domain<T>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
atlas::host_shared_ptr<Domain<T>>
Domain<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<Domain<T>>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {

    auto op       = geometry->make_geometry_operator();
    auto bound    = op.bound();
    _lower_corner = bound.lower_corner;
    _upper_corner = bound.upper_corner;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_lower_corner(const Vector3<T>& v) noexcept {

    _lower_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_upper_corner(const Vector3<T>& v) noexcept {

    _upper_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_cell_size(T h) noexcept {

    _cell_size = h;
    return *this;
}

template <typename T>
void
Domain<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Domain::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(
        _upper_corner.x > _lower_corner.x && _upper_corner.y > _lower_corner.y && _upper_corner.z > _lower_corner.z)
        << "Domain::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const T inv_h         = T(1) / _cell_size;
    const Vector3<int> gs = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * inv_h) + Vector3<T> { T(1), T(1), T(1) });

    atlas::check<std::invalid_argument>(gs.x >= 1 && gs.y >= 1 && gs.z >= 1)
        << "Domain::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Domain::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const long long cells64 = nx * ny * nz;

    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Domain::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

}