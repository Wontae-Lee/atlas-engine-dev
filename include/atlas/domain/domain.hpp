#pragma once
#include <atlas/logging/logging.h>

namespace atlas::system {

// ---------------------------------
// Domain<T>
// ---------------------------------

template <typename T>
Domain<T>::Domain(const Vector3<T>& lower_corner,
                  const Vector3<T>& upper_corner,
                  T cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

    // ------------------------------------------------------------
    // Constructs a simulation domain defined by:
    //   - lower_corner: minimum coordinates of the domain (inclusive)
    //   - upper_corner: maximum coordinates of the domain (inclusive)
    //   - cell_size   : uniform grid spacing h (> 0)
    //
    // Side effects:
    //   - Logs construction parameters (if logging is enabled).
    //
    // Postconditions:
    //   - _cell_volume and _inv_h are computed.
    //   - _grid_size is computed from bounds and cell size.
    //   - Device buffers are allocated and initialized.
    // ------------------------------------------------------------
    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    // ------------------------------------------------------------
    // Precompute derived quantities:
    //   - cell volume (h^3)
    //   - inverse cell size (1/h) for repeated scaling
    // ------------------------------------------------------------
    _cell_volume = _cell_size * _cell_size * _cell_size;
    _inv_h       = T(1) / _cell_size;

    // ------------------------------------------------------------
    // Compute discrete grid resolution:
    //   grid_size = floor((upper - lower) / h) + 1
    //
    // Rationale:
    //   - The +1 accounts for including both ends of the interval.
    //   - floor ensures a conservative discretization within bounds.
    //
    // Note:
    //   - cast_to<int> converts the floating result to integer grid
    //     dimensions.
    // ------------------------------------------------------------
    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    // ------------------------------------------------------------
    // Total number of cells in the grid.
    // ------------------------------------------------------------
    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;

    // ------------------------------------------------------------
    // Allocate and initialize device-side storage.
    //
    // Initialization:
    //   - temperature initialized to 0
    //   - field_force initialized to (0,0,0)
    // ------------------------------------------------------------
    d_temperature.resize(_num_of_cells, T(0));
    d_field_force.resize(_num_of_cells, Vector3<T> { T(0), T(0), T(0) });
}

template <typename T>
typename Domain<T>::Builder
Domain<T>::builder() noexcept {
    // ------------------------------------------------------------
    // Returns a default-initialized Builder for Domain<T>.
    //
    // Notes:
    //   - The builder collects parameters (bounds, cell size) and
    //     validates them before constructing a Domain instance.
    //   - This function is noexcept because it only returns a value
    //     with no allocations or validation.
    // ------------------------------------------------------------
    return Builder {};
}

template <typename T>
DomainDeviceProbe<T>
Domain<T>::make_device_probe() noexcept {
    // ------------------------------------------------------------
    // Creates a lightweight "device probe" struct containing:
    //   - raw pointers to device buffers
    //   - immutable domain metadata (bounds, grid, spacing)
    //
    // Intended usage:
    //   - Pass to CUDA kernels / device code without copying
    //     heavy owning containers.
    //
    // Safety:
    //   - Returned pointers remain valid as long as the Domain
    //     instance owns the underlying buffers and they are not
    //     reallocated.
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Increment probe count.
    // ------------------------------------------------------------
    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    DomainDeviceProbe<T> probe;

    // ------------------------------------------------------------
    // Extract raw device pointers from owning device containers.
    // ------------------------------------------------------------
    probe.temperature = atlas::raw_pointer_cast(d_temperature.data());
    probe.field_force = atlas::raw_pointer_cast(d_field_force.data());

    // ------------------------------------------------------------
    // Copy domain geometry and discretization metadata.
    // ------------------------------------------------------------
    probe.lower_corner = _lower_corner;
    probe.upper_corner = _upper_corner;
    probe.grid_size    = _grid_size;

    // ------------------------------------------------------------
    // Copy derived grid quantities.
    // ------------------------------------------------------------
    probe.cell_size   = _cell_size;
    probe.cell_volume = _cell_volume;
    probe.inv_h       = _inv_h;

    // ------------------------------------------------------------
    // Copy total cell count.
    // ------------------------------------------------------------
    probe.num_of_cells = _num_of_cells;

    return probe;
}

template <typename T>
int
Domain<T>::number_of_cells() const noexcept {
    // ------------------------------------------------------------
    // Returns the total number of cells in the grid.
    // ------------------------------------------------------------
    return _num_of_cells;
}

template <typename T>
Vector3<T>
Domain<T>::lower_corner() const noexcept {
    // ------------------------------------------------------------
    // Returns the lower (minimum) corner of the domain.
    // ------------------------------------------------------------
    return _lower_corner;
}

template <typename T>
Vector3<T>
Domain<T>::upper_corner() const noexcept {
    // ------------------------------------------------------------
    // Returns the upper (maximum) corner of the domain.
    // ------------------------------------------------------------
    return _upper_corner;
}

template <typename T>
Vector3<int>
Domain<T>::grid_size() const noexcept {
    // ------------------------------------------------------------
    // Returns the integer grid resolution (cells per axis).
    // ------------------------------------------------------------
    return _grid_size;
}

template <typename T>
T
Domain<T>::cell_size() const noexcept {
    // ------------------------------------------------------------
    // Returns the uniform cell size (grid spacing) of the domain.
    // ------------------------------------------------------------
    return _cell_size;
}

template <typename T>
T
Domain<T>::cell_volume() const noexcept {
    // ------------------------------------------------------------
    // Returns the cell volume (h^3) of the domain.
    // ------------------------------------------------------------
    return _cell_volume;
}

template <typename T>
T
Domain<T>::inverse_cell_size() const noexcept {
    // ------------------------------------------------------------
    // Returns the inverse cell size (1/h) of the domain.
    // ------------------------------------------------------------
    return _inv_h;
}

// ---------------------------------
// Builder
// ---------------------------------
template <typename T>
Domain<T>
Domain<T>::Builder::build() const {
    // ------------------------------------------------------------
    // Validates inputs and returns a fully constructed Domain<T>.
    // ------------------------------------------------------------
    validate();
    return Domain<T>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
atlas::host_shared_ptr<Domain<T>>
Domain<T>::Builder::make_host_shared() const {
    // ------------------------------------------------------------
    // Validates inputs and returns a host_shared_ptr owning the
    // constructed Domain<T>.
    // ------------------------------------------------------------
    validate();
    return atlas::make_host_shared<Domain<T>>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {
    // ------------------------------------------------------------
    // Configures the builder's bounds from a geometry object.
    // ------------------------------------------------------------
    auto op       = geometry->make_geometry_operator();
    auto bound    = op.bound();
    _lower_corner = bound.lower_corner;
    _upper_corner = bound.upper_corner;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_lower_corner(const Vector3<T>& v) noexcept {
    // ------------------------------------------------------------
    // Sets the lower (minimum) corner of the domain.
    // ------------------------------------------------------------
    _lower_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_upper_corner(const Vector3<T>& v) noexcept {
    // ------------------------------------------------------------
    // Sets the upper (maximum) corner of the domain.
    // ------------------------------------------------------------
    _upper_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_cell_size(T h) noexcept {
    // ------------------------------------------------------------
    // Sets the uniform cell size (grid spacing) of the domain.
    // Must be > 0 (validated in validate()).
    // ------------------------------------------------------------
    _cell_size = h;
    return *this;
}

template <typename T>
void
Domain<T>::Builder::validate() const {
    // ------------------------------------------------------------
    // Validates builder parameters and throws std::invalid_argument
    // on failure.
    //
    // Validation rules:
    //   1) cell_size must be positive
    //   2) upper_corner must be strictly greater than lower_corner
    //      on all axes
    //   3) computed grid_size must be >= 1 on all axes
    //   4) total cell count must fit in int (prevent overflow)
    //
    // Notes:
    //   - Uses atlas::check which throws on failure and can also
    //     emit an error log line if logging is enabled.
    // ------------------------------------------------------------

    // 1) cell_size must be positive
    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Domain::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    // 2) upper_corner must be strictly greater than lower_corner on all axes
    atlas::check<std::invalid_argument>(
        _upper_corner.x > _lower_corner.x && _upper_corner.y > _lower_corner.y && _upper_corner.z > _lower_corner.z)
        << "Domain::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    // 3) computed grid_size must be >= 1
    //
    // Compute:
    //   grid_size = floor((upper - lower) / h) + 1
    const T inv_h         = T(1) / _cell_size;
    const Vector3<int> gs = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * inv_h) + Vector3<T> { T(1), T(1), T(1) });

    atlas::check<std::invalid_argument>(gs.x >= 1 && gs.y >= 1 && gs.z >= 1)
        << "Domain::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    // 4) total cell count must fit in int
    //
    // Promote to 64-bit to avoid intermediate overflow.
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

} // namespace atlas::system
