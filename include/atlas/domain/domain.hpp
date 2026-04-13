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

    // Log the domain configuration at construction time so the runtime can
    // clearly report the spatial bounds and discretization that define the grid.
    //
    // This is useful for debugging setup issues such as:
    // - incorrect bounding corners,
    // - unexpected cell resolution,
    // - mismatched domain size relative to the intended simulation setup.
    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    // Precompute the volume of one cubic cell.
    //
    // Since the domain currently uses a uniform Cartesian grid with identical
    // spacing on all three axes, cell volume is simply h^3.
    //
    // Caching this once avoids recomputing the same expression in downstream
    // stages such as measurement or solver logic.
    _cell_volume = _cell_size * _cell_size * _cell_size;

    // Precompute the reciprocal of the cell size.
    //
    // This value is commonly used when converting world-space coordinates into
    // grid-space coordinates, so storing it once is cheaper than repeated division.
    _inv_h = T(1) / _cell_size;

    // Compute the integer grid resolution along each axis.
    //
    // Interpretation:
    // - `_upper_corner - _lower_corner` gives the world-space extent.
    // - multiplying by `_inv_h` converts that extent into cell units.
    // - `floor(...)` ensures we stay within the covered range in integer grid space.
    // - `+ 1` makes the discretization inclusive of both lower and upper bounds.
    //
    // The inclusive convention means that even a span exactly equal to one cell size
    // produces two grid points / one offset-inclusive index span.
    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    // Compute the total number of cells in the structured grid.
    //
    // This value is later used for:
    // - device buffer allocation,
    // - runtime probe metadata,
    // - domain-wide iteration and indexing.
    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;

    // Choose the initial value for the temperature field.
    //
    // Rule:
    // - if the domain is isothermal and an isothermal temperature has been provided,
    //   initialize every cell with that prescribed temperature;
    // - otherwise, initialize the field to zero.
    //
    // This gives isothermal domains a fully populated thermal field immediately,
    // while non-isothermal domains begin from a neutral default.
    const T initial_temperature = (_type == DomainType::isothermal && _isothermal_field_temperature.has_value())
        ? *_isothermal_field_temperature
        : T(0);

    // Allocate the per-cell temperature field on the device and initialize every
    // entry to the chosen starting temperature.
    d_field_temperature.resize(_num_of_cells, initial_temperature);
}

template <typename T>
typename Domain<T>::Builder
Domain<T>::builder() noexcept {
    // Return a fresh builder object for staged `Domain<T>` construction.
    //
    // This path is useful when domain parameters such as bounds, cell size,
    // type, and optional temperature are configured progressively.
    return Builder {};
}

template <typename T>
DomainDeviceProbe<T>
Domain<T>::make_device_probe() noexcept {

    // Track how many times a device probe has been requested from this domain.
    //
    // The surrounding runtime is designed around one authoritative probe
    // instance that is created and owned by the system.
    ++_probe_count;

    // Enforce the invariant that only one domain probe should exist.
    //
    // Rationale:
    // - downstream runtime code expects a single canonical device-side view,
    // - multiple probes could imply duplicated ownership assumptions or
    //   inconsistent runtime binding.
    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Create the probe object that will expose domain state to backend/device code.
    DomainDeviceProbe<T> probe;

    // Export only the low-level device-visible data needed by kernels.
    //
    // Exposed contents:
    // - domain type,
    // - raw pointer to temperature field,
    // - optional raw pointer to force field,
    // - geometric bounds and grid metadata,
    // - cell metrics and total cell count.
    //
    // The optional force field pointer is set to `nullptr` when no force field
    // has been configured.
    probe.type              = _type;
    probe.field_temperature = atlas::raw_pointer_cast(d_field_temperature.data());
    probe.field_force       = d_field_force.empty() ? nullptr : atlas::raw_pointer_cast(d_field_force.data());

    // Publish geometric domain extents and discrete grid shape.
    probe.lower_corner = _lower_corner;
    probe.upper_corner = _upper_corner;
    probe.grid_size    = _grid_size;

    // Publish precomputed cell metrics so backend code can avoid recomputing them.
    probe.cell_size   = _cell_size;
    probe.cell_volume = _cell_volume;
    probe.inv_h       = _inv_h;

    // Publish total number of grid cells for flat indexing and bounds checks.
    probe.num_of_cells = _num_of_cells;

    return probe;
}

template <typename T>
int
Domain<T>::number_of_cells() const noexcept {
    // Return the total number of cells in the domain grid.
    return _num_of_cells;
}

template <typename T>
DomainType
Domain<T>::type() const noexcept {
    // Return the configured domain type.
    return _type;
}

template <typename T>
T
Domain<T>::isothermal_field_temperature() const {
    // This query is only meaningful for isothermal domains.
    //
    // Non-isothermal domains do not have a single prescribed temperature that
    // semantically represents the whole field.
    if (_type != DomainType::isothermal) {
        atlas::logger::error()
            << "Domain: isothermal_field_temperature() is only valid for isothermal domains.";
        throw std::runtime_error("Domain: isothermal_field_temperature() is only valid for isothermal domains.");
    }

    // Even for an isothermal domain, the configuration must actually contain
    // a stored prescribed temperature value.
    if (!_isothermal_field_temperature.has_value()) {
        atlas::logger::error()
            << "Domain: isothermal_field_temperature() is not configured for this isothermal domain.";
        throw std::runtime_error("Domain: isothermal_field_temperature() is not configured for this isothermal domain.");
    }

    // Return the configured uniform field temperature.
    return *_isothermal_field_temperature;
}

template <typename T>
void
Domain<T>::set_field_force(DeviceBuffer<Vector3<T>> field_force) noexcept {
    // Replace the optional per-cell force field in a single move operation.
    //
    // This allows callers to install prebuilt device-resident force data
    // without copying individual entries.
    d_field_force = std::move(field_force);
}

template <typename T>
DeviceBuffer<Vector3<T>>&
Domain<T>::field_force() noexcept {
    // Return mutable access to the optional device-resident per-cell force field.
    return d_field_force;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
Domain<T>::field_force() const noexcept {
    // Return read-only access to the optional device-resident per-cell force field.
    return d_field_force;
}

template <typename T>
Vector3<T>
Domain<T>::lower_corner() const noexcept {
    // Return the domain's lower spatial bound.
    return _lower_corner;
}

template <typename T>
Vector3<T>
Domain<T>::upper_corner() const noexcept {
    // Return the domain's upper spatial bound.
    return _upper_corner;
}

template <typename T>
Vector3<int>
Domain<T>::grid_size() const noexcept {
    // Return the discrete grid resolution along x/y/z.
    return _grid_size;
}

template <typename T>
T
Domain<T>::cell_size() const noexcept {
    // Return the uniform cell edge length.
    return _cell_size;
}

template <typename T>
T
Domain<T>::cell_volume() const noexcept {
    // Return the precomputed volume of one grid cell.
    return _cell_volume;
}

template <typename T>
T
Domain<T>::inverse_cell_size() const noexcept {
    // Return the precomputed reciprocal cell size.
    return _inv_h;
}

template <typename T>
Domain<T>
Domain<T>::Builder::build() const {
    // Validate the staged builder parameters before constructing the domain.
    validate();

    // Construct and return the domain by value using the validated builder state.
    return Domain<T>(_lower_corner, _upper_corner, _cell_size, _type, _isothermal_field_temperature);
}

template <typename T>
atlas::host_shared_ptr<Domain<T>>
Domain<T>::Builder::make_host_shared() const {
    // Validate the staged builder parameters before constructing the domain
    // in host-shared managed storage.
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
    // Use the supplied geometry's axis-aligned bounding box to initialize
    // the domain bounds automatically.
    //
    // This is a convenience path for constructing a domain that tightly encloses
    // a geometry without manually specifying corners.
    auto op       = geometry->make_geometry_operator();
    auto bound    = op.bound();
    _lower_corner = bound.lower_corner;
    _upper_corner = bound.upper_corner;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_lower_corner(const Vector3<T>& v) noexcept {
    // Store the lower domain corner in the builder.
    _lower_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_upper_corner(const Vector3<T>& v) noexcept {
    // Store the upper domain corner in the builder.
    _upper_corner = v;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_cell_size(T h) noexcept {
    // Store the uniform grid cell size in the builder.
    _cell_size = h;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_type(const DomainType type) noexcept {
    // Emit a warning when the domain is configured as isothermal.
    //
    // This highlights a semantic restriction in the wider runtime:
    // temperature checks or temperature updates driven by Measure are not
    // allowed for isothermal domains because the thermal state is prescribed.
    if (type == DomainType::isothermal) {
        atlas::logger::warn()
            << "Domain::Builder: isothermal domain selected. "
            << "Temperature checks or updates driven by Measure are forbidden for this domain type.";
    }

    // Store the selected domain type in the builder.
    _type = type;
    return *this;
}

template <typename T>
typename Domain<T>::Builder&
Domain<T>::Builder::with_temperature(const T temperature) noexcept {
    // Store the optional prescribed temperature used by isothermal domains.
    _isothermal_field_temperature = temperature;
    return *this;
}

template <typename T>
void
Domain<T>::Builder::validate() const {
    // The cell size must be strictly positive.
    //
    // A non-positive cell size would make:
    // - inverse cell size undefined or non-physical,
    // - grid sizing invalid,
    // - cell metrics meaningless.
    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Domain::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    // Require strictly increasing bounds on every axis.
    //
    // This enforces a non-degenerate positive-volume domain extent in x, y, and z.
    atlas::check<std::invalid_argument>(
        _upper_corner.x > _lower_corner.x && _upper_corner.y > _lower_corner.y && _upper_corner.z > _lower_corner.z)
        << "Domain::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    // Reproduce the constructor's grid sizing logic here so invalid grid geometry
    // is detected before any runtime allocation occurs.
    const T inv_h = T(1) / _cell_size;

    // Compute the candidate grid size using the same inclusive-bounds rule as
    // the constructor.
    const Vector3<int> gs = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * inv_h) + Vector3<T> { T(1), T(1), T(1) });

    // Every axis must produce at least one valid cell index extent.
    atlas::check<std::invalid_argument>(gs.x >= 1 && gs.y >= 1 && gs.z >= 1)
        << "Domain::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    // Promote grid dimensions to a wider integer type before computing the total
    // number of cells so overflow can be checked safely.
    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    // Guard against invalid promoted values as an additional consistency check.
    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Domain::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    // Compute total cell count in widened precision.
    const long long cells64 = nx * ny * nz;

    // Ensure the total number of cells is:
    // - positive,
    // - representable in the `int`-based runtime/device probe.
    //
    // This protects later code paths that rely on `int` cell counts and flat indexing.
    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Domain::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    if (_type == DomainType::isothermal) {
        // Isothermal domains must provide an explicit prescribed temperature.
        atlas::check<std::invalid_argument>(_isothermal_field_temperature.has_value())
            << "Domain::Builder validation failed: isothermal domain temperature must be provided.";

        // The prescribed isothermal temperature must be finite and non-negative.
        //
        // This enforces a physically meaningful stored state and avoids propagating
        // invalid floating-point values into the initialized temperature field.
        atlas::check<std::invalid_argument>(
            std::isfinite(*_isothermal_field_temperature) && *_isothermal_field_temperature >= T(0))
            << "Domain::Builder validation failed: isothermal domain temperature must be finite and non-negative. "
            << "temperature=" << *_isothermal_field_temperature;
    }
}

} // namespace atlas::system