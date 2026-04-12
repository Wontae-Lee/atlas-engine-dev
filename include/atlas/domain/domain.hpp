#pragma once
#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
Domain<T>::Domain(const Vector3<T>& lower_corner,
                  const Vector3<T>& upper_corner,
                  T cell_size,
                  const DomainType type,
                  std::optional<T> temperature)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size)
    , _type(type)
    , _isothermal_field_temperature(std::move(temperature)) {

    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    // Cache geometric invariants once so runtime stages can reuse them cheaply.
    _cell_volume = _cell_size * _cell_size * _cell_size;
    _inv_h       = T(1) / _cell_size;

    // The grid is inclusive of both bounds, so each axis gets a +1 cell offset.
    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;

    // Isothermal domains start prefilled with their prescribed temperature.
    const T initial_temperature = (_type == DomainType::isothermal && _isothermal_field_temperature.has_value())
        ? *_isothermal_field_temperature
        : T(0);

    d_field_temperature.resize(_num_of_cells, initial_temperature);
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

    // System owns the single authoritative domain probe for the runtime.
    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    DomainDeviceProbe<T> probe;

    // Only raw device pointers and POD metadata are exported to backend code.
    probe.type              = _type;
    probe.field_temperature = atlas::raw_pointer_cast(d_field_temperature.data());
    probe.field_force       = d_field_force.empty() ? nullptr : atlas::raw_pointer_cast(d_field_force.data());

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
DomainType
Domain<T>::type() const noexcept {

    return _type;
}

template <typename T>
T
Domain<T>::isothermal_field_temperature() const {
    if (_type != DomainType::isothermal) {
        atlas::logger::error()
            << "Domain: isothermal_field_temperature() is only valid for isothermal domains.";
        throw std::runtime_error("Domain: isothermal_field_temperature() is only valid for isothermal domains.");
    }

    if (!_isothermal_field_temperature.has_value()) {
        atlas::logger::error()
            << "Domain: isothermal_field_temperature() is not configured for this isothermal domain.";
        throw std::runtime_error("Domain: isothermal_field_temperature() is not configured for this isothermal domain.");
    }

    return *_isothermal_field_temperature;
}

template <typename T>
void
Domain<T>::set_field_force(DeviceBuffer<Vector3<T>> field_force) noexcept {

    // Replace the optional per-cell force field in one move.
    d_field_force = std::move(field_force);
}


template <typename T>
DeviceBuffer<Vector3<T>>&
Domain<T>::field_force() noexcept {

    return d_field_force;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
Domain<T>::field_force() const noexcept {

    return d_field_force;
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
    return Domain<T>(_lower_corner, _upper_corner, _cell_size, _type, _isothermal_field_temperature);
}

template <typename T>
atlas::host_shared_ptr<Domain<T>>
Domain<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<Domain<T>>(
        _lower_corner,
        _upper_corner,
        _cell_size,
        _type,
        _isothermal_field_temperature);
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {

    // Reuse the geometry operator's bounding box to seed domain bounds.
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
typename Domain<T>::Builder&
Domain<T>::Builder::with_type(const DomainType type) noexcept {

    if (type == DomainType::isothermal) {
        atlas::logger::warn()
            << "Domain::Builder: isothermal domain selected. "
            << "Temperature checks or updates driven by Measure are forbidden for this domain type.";
    }

    _type = type;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_temperature(const T temperature) noexcept {

    _isothermal_field_temperature = temperature;
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
    // Mirror constructor grid sizing to catch invalid bounds before allocation.
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

    // Keep total cell count representable by the int-based runtime probe.
    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Domain::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    if (_type == DomainType::isothermal) {
        atlas::check<std::invalid_argument>(_isothermal_field_temperature.has_value())
            << "Domain::Builder validation failed: isothermal domain temperature must be provided.";

        atlas::check<std::invalid_argument>(
            std::isfinite(*_isothermal_field_temperature) && *_isothermal_field_temperature >= T(0))
            << "Domain::Builder validation failed: isothermal domain temperature must be finite and non-negative. "
            << "temperature=" << *_isothermal_field_temperature;
    }
}

}
