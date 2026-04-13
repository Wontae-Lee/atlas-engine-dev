#pragma once

/**
 * @file domain.h
 * @brief Declares the simulation domain, its device-facing probe, and the builder API.
 *
 * @details
 * This header defines @ref atlas::system::Domain, the runtime object that owns
 * the structured simulation domain used by Atlas subsystems such as search,
 * measurement, encoding, and solver execution.
 *
 * A domain combines:
 * - geometric bounds in world space,
 * - a regular Cartesian grid discretization,
 * - grid-aligned field storage such as temperature and optional force fields,
 * - precomputed metadata for fast coordinate-to-cell mapping,
 * - a lightweight device-facing probe for backend execution.
 *
 * ## Role in the simulation pipeline
 * The domain acts as the canonical source of grid-based environmental and field
 * information. It is commonly consumed by:
 * - spatial search stages for cell lookup and neighborhood organization,
 * - codec stages for field sampling or per-cell reduced representations,
 * - measure stages for computing diagnostics over a structured grid,
 * - solver stages for applying domain-aligned forces or thermal conditions.
 *
 * ## Discretization model
 * The domain is defined by:
 * - a lower world-space corner,
 * - an upper world-space corner,
 * - a uniform cubic cell size.
 *
 * These values determine a regular Cartesian grid with:
 * - integer grid resolution along each axis,
 * - a total number of cells,
 * - a precomputed cell volume,
 * - a precomputed inverse cell size for fast indexing.
 *
 * ## Temperature modes
 * The domain supports two temperature interpretation modes:
 * - @ref DomainType::isothermal, where the field temperature is prescribed,
 * - @ref DomainType::variable, where the field temperature may evolve spatially
 *   and temporally according to the simulation.
 *
 * ## Device probe model
 * Since backend kernels and parallel lambdas should not carry host-side ownership
 * structures, the domain exposes a compact @ref DomainDeviceProbe containing:
 * - raw pointers to device-resident field buffers,
 * - geometric bounds,
 * - grid resolution and spacing metadata,
 * - cached values needed for efficient cell mapping.
 *
 * The system runtime typically owns the single authoritative probe instance for
 * a live simulation and reuses it across stages.
 *
 * ## Construction
 * A domain is intentionally not default-constructible. It must be created with
 * explicit geometric and discretization information, either:
 * - directly through the constructor, or
 * - through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for domain geometry and field values.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <optional>

namespace atlas::system {

/**
 * @brief Selects how the domain temperature field should be interpreted.
 *
 * @details
 * This enum specifies whether the domain temperature is treated as:
 * - a fixed prescribed field, or
 * - a variable field that may be updated during the simulation.
 *
 * The chosen mode affects how the domain is initialized and how temperature
 * data is expected to be consumed by downstream runtime components.
 */
enum class DomainType : int {
    isothermal, ///< Temperature is prescribed and fixed according to the domain configuration.
    variable    ///< Temperature is stored as an evolvable field and may vary during the simulation.
};

/**
 * @brief Lightweight device-facing view of domain state.
 *
 * @details
 * @ref DomainDeviceProbe is the compact runtime structure passed to device kernels
 * and backend-parallel execution paths.
 *
 * It exists to expose the essential domain data needed during low-level execution
 * without carrying host-side ownership, polymorphism, or heavyweight containers.
 *
 * The probe typically provides:
 * - raw pointers to device-resident field buffers,
 * - world-space domain bounds,
 * - integer grid dimensions,
 * - cached spacing-related quantities for efficient coordinate mapping.
 *
 * ## Typical usage
 * This probe is commonly consumed by code that needs to:
 * - determine whether a point lies within the domain,
 * - map a world-space position to a cell index,
 * - sample or update temperature and force fields,
 * - iterate over domain-aligned grid data in backend code.
 *
 * ## Ownership
 * All pointers stored in the probe are non-owning. Their lifetime must remain
 * valid for the duration of any backend work using the probe.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct DomainDeviceProbe {
    /**
     * @brief Runtime domain temperature mode.
     *
     * @details
     * Indicates whether the domain should be interpreted as isothermal or variable.
     */
    DomainType type = DomainType::variable;

    /**
     * @brief Pointer to the per-cell temperature field.
     *
     * @details
     * Points to device-resident storage containing one temperature value per cell.
     * For isothermal domains, this field may still be uniformly populated with the
     * prescribed temperature value.
     */
    T* field_temperature = nullptr;

    /**
     * @brief Pointer to the optional per-cell external force field.
     *
     * @details
     * Points to device-resident storage containing one force vector per cell.
     * When no force field is configured, this pointer may be null or refer to an
     * empty buffer depending on implementation policy.
     */
    Vector3<T>* field_force = nullptr;

    /**
     * @brief World-space minimum corner of the domain bounds.
     *
     * @details
     * Defines the lower Cartesian bound of the discretized domain.
     */
    Vector3<T> lower_corner;

    /**
     * @brief World-space maximum corner of the domain bounds.
     *
     * @details
     * Defines the upper Cartesian bound of the discretized domain.
     */
    Vector3<T> upper_corner;

    /**
     * @brief Number of grid cells along each Cartesian axis.
     *
     * @details
     * Stores the discrete grid resolution in x, y, and z.
     */
    Vector3<int> grid_size { 1, 1, 1 };

    /**
     * @brief Uniform cubic cell spacing.
     *
     * @details
     * Represents the edge length of each regular grid cell.
     */
    T cell_size = T(1);

    /**
     * @brief Precomputed volume of a single grid cell.
     *
     * @details
     * Since the grid is uniform and cubic, this is typically equal to
     * \f$h^3\f$, where \f$h\f$ is @ref cell_size.
     */
    T cell_volume = T(1);

    /**
     * @brief Reciprocal of the cell size.
     *
     * @details
     * Cached for fast world-space coordinate to cell-index conversion.
     */
    T inv_h = T(1);

    /**
     * @brief Total number of cells in the domain.
     *
     * @details
     * Typically equal to:
     * \f[
     * n_x \times n_y \times n_z
     * \f]
     * where \f$(n_x,n_y,n_z)\f$ is @ref grid_size.
     */
    int num_of_cells = 0;
};

/**
 * @brief Owns the simulation grid, field storage, and domain metadata.
 *
 * @details
 * @ref Domain is the host-side runtime object representing a discretized
 * Cartesian simulation region together with its associated field storage.
 *
 * It owns:
 * - device-resident per-cell temperature values,
 * - optional device-resident per-cell force values,
 * - geometric bounds defining the world-space extent,
 * - grid discretization metadata derived from the chosen cell size,
 * - bookkeeping needed to construct device probes.
 *
 * ## Geometric interpretation
 * The domain is defined by an axis-aligned box in world space:
 * - @ref _lower_corner gives the minimum bound,
 * - @ref _upper_corner gives the maximum bound.
 *
 * These bounds are discretized into a uniform Cartesian grid using the specified
 * cell size.
 *
 * ## Field storage
 * The domain currently supports:
 * - a temperature field with one scalar value per cell,
 * - an optional force field with one vector value per cell.
 *
 * These fields are stored in device-resident buffers so they can be consumed
 * efficiently by backend execution code.
 *
 * ## Temperature semantics
 * The domain mode controls how temperature values should be interpreted:
 * - in @ref DomainType::isothermal mode, the domain has a prescribed temperature,
 * - in @ref DomainType::variable mode, temperature is treated as a general field.
 *
 * ## Probe creation
 * The domain can produce a @ref DomainDeviceProbe through @ref make_device_probe.
 * The runtime system typically caches that probe as the single authoritative
 * device-facing domain view.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Domain {
public:
    /**
     * @brief Fluent builder used to configure and construct @ref Domain instances.
     *
     * @details
     * The builder stages bounds, discretization, and temperature-mode parameters,
     * validates them, and then materializes either:
     * - a value instance of @ref Domain, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

    /**
     * @brief Deleted default constructor.
     *
     * @details
     * A domain must be created with explicit geometric bounds and discretization
     * parameters, so default construction is disallowed.
     */
    Domain() = delete;

    /**
     * @brief Construct a domain from explicit bounds and discretization parameters.
     *
     * @details
     * Initializes the world-space bounds, derives the regular grid resolution from
     * the supplied cell size, allocates field storage, and configures the domain's
     * temperature interpretation mode.
     *
     * @param lower_corner Minimum world-space bound of the domain.
     * @param upper_corner Maximum world-space bound of the domain.
     * @param cell_size Uniform cubic cell spacing.
     * @param type Domain temperature mode.
     * @param temperature Optional prescribed temperature for isothermal domains.
     *
     * @note
     * The exact initialization policy for the per-cell temperature field is
     * implementation-defined in `domain.hpp`.
     */
    Domain(const Vector3<T>& lower_corner,
           const Vector3<T>& upper_corner,
           T cell_size,
           DomainType type              = DomainType::variable,
           std::optional<T> temperature = std::nullopt);

    /**
     * @brief Default destructor.
     *
     * @details
     * Since resources are owned through RAII-managed member objects, the destructor
     * is defaulted.
     */
    ~Domain() = default;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder for fluent domain construction.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Create the device probe consumed by runtime kernels.
     *
     * @details
     * Builds and returns a compact @ref DomainDeviceProbe containing raw pointers
     * and precomputed grid metadata derived from this domain instance.
     *
     * The system runtime typically owns the single authoritative probe for a live
     * simulation and reuses it across multiple stages.
     *
     * @return Device-facing probe referencing this domain's runtime state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Return the total number of grid cells.
     *
     * @details
     * Returns the cached total number of discretized cells in the domain.
     *
     * @return Total number of cells.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    /**
     * @brief Return the configured domain temperature mode.
     *
     * @return Current @ref DomainType value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DomainType
    type() const noexcept;

    /**
     * @brief Return the prescribed field temperature for isothermal domains.
     *
     * @details
     * Retrieves the stored isothermal temperature value when the domain is in
     * @ref DomainType::isothermal mode.
     *
     * @return Prescribed isothermal field temperature.
     *
     * @throws std::runtime_error
     * Thrown if the domain is not isothermal or if no prescribed temperature
     * was configured.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    isothermal_field_temperature() const;

    /**
     * @brief Replace the optional field-force buffer.
     *
     * @details
     * Installs a new device-resident per-cell force field for the domain.
     *
     * @param field_force Device buffer storing one force vector per cell.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_field_force(DeviceBuffer<Vector3<T>> field_force) noexcept;

    /**
     * @brief Return mutable access to the field-force buffer.
     *
     * @return Mutable reference to the device-resident force field buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    field_force() noexcept;

    /**
     * @brief Return const access to the field-force buffer.
     *
     * @return Const reference to the device-resident force field buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Vector3<T>>&
    field_force() const noexcept;

    /**
     * @brief Return the lower world-space bound of the domain.
     *
     * @return Lower Cartesian bound.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Return the upper world-space bound of the domain.
     *
     * @return Upper Cartesian bound.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    /**
     * @brief Return the integer grid resolution.
     *
     * @details
     * Returns the number of cells along each Cartesian axis.
     *
     * @return Grid resolution as `(nx, ny, nz)`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Return the uniform cell size.
     *
     * @return Cubic cell edge length.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    /**
     * @brief Return the volume of a single grid cell.
     *
     * @return Cached cubic cell volume.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    /**
     * @brief Return the reciprocal of the uniform cell size.
     *
     * @return Cached value of `1 / cell_size`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

private:
    /**
     * @brief Device-resident per-cell temperature field.
     *
     * @details
     * Stores one scalar temperature value per domain cell.
     */
    DeviceBuffer<T> d_field_temperature;

    /**
     * @brief Optional device-resident per-cell force field.
     *
     * @details
     * Stores one force vector per domain cell when such a field is configured.
     */
    DeviceBuffer<Vector3<T>> d_field_force;

    /**
     * @brief Minimum world-space domain bound.
     *
     * @details
     * Lower Cartesian corner of the domain.
     */
    Vector3<T> _lower_corner;

    /**
     * @brief Maximum world-space domain bound.
     *
     * @details
     * Upper Cartesian corner of the domain.
     */
    Vector3<T> _upper_corner;

    /**
     * @brief Number of cells along x, y, and z.
     *
     * @details
     * Cached regular-grid resolution derived from the bounds and cell size.
     */
    Vector3<int> _grid_size { 1, 1, 1 };

    /**
     * @brief Uniform cubic cell spacing.
     *
     * @details
     * Edge length of each cell in the Cartesian discretization.
     */
    T _cell_size = T(1);

    /**
     * @brief Cached cell volume.
     *
     * @details
     * Typically equal to \f$h^3\f$, where \f$h\f$ is @ref _cell_size.
     */
    T _cell_volume = T(1);

    /**
     * @brief Cached reciprocal cell size.
     *
     * @details
     * Used for fast coordinate-to-index conversion.
     */
    T _inv_h = T(1);

    /**
     * @brief Cached total number of cells.
     *
     * @details
     * Equal to the product of the grid dimensions.
     */
    int _num_of_cells = 1;

    /**
     * @brief Domain temperature mode.
     *
     * @details
     * Indicates whether the domain is isothermal or variable-temperature.
     */
    DomainType _type = DomainType::variable;

    /**
     * @brief Prescribed field temperature for isothermal mode.
     *
     * @details
     * Present only when the domain is configured to use a fixed isothermal
     * temperature.
     */
    std::optional<T> _isothermal_field_temperature;

    /**
     * @brief Internal counter tracking probe creation.
     *
     * @details
     * May be used by the implementation to monitor probe refresh activity.
     */
    std::uint64_t _probe_count = 0;
};

/**
 * @brief Fluent builder for @ref Domain.
 *
 * @details
 * This builder provides a controlled way to construct a @ref Domain while
 * staging bounds, discretization, and temperature-related configuration.
 *
 * ## Typical usage
 * @code
 * auto domain = atlas::Domain<float>::builder()
 *     .with_lower_corner({0.0f, 0.0f, 0.0f})
 *     .with_upper_corner({1.0f, 1.0f, 1.0f})
 *     .with_cell_size(0.05f)
 *     .with_type(atlas::DomainType::variable)
 *     .build();
 * @endcode
 *
 * The builder may also derive bounds from a geometry object's bounding box.
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - lower corner is component-wise less than or equal to upper corner,
 * - cell size is positive and finite,
 * - isothermal temperature is provided when required,
 * - staged parameters are internally consistent.
 *
 * The exact validation rules are implementation-defined in `domain.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Domain<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with unit-cube bounds, unit cell size, variable
     * temperature mode, and no prescribed isothermal temperature.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Domain by value after validation.
     *
     * @details
     * Validates the staged builder state and constructs the final domain object.
     *
     * @return Fully constructed domain value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Domain<T>
    build() const;

    /**
     * @brief Build a configured @ref Domain in a host_shared_ptr after validation.
     *
     * @details
     * Validates the staged builder state, constructs the domain, and returns it
     * in a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<Domain<T>>` owning the constructed domain.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Domain<T>>
    make_host_shared() const;

    /**
     * @brief Derive the domain bounds from a geometry object's bounding box.
     *
     * @details
     * Extracts the axis-aligned bounding box of the supplied geometry and uses
     * it to populate the staged domain bounds.
     *
     * @param geometry Geometry providing the source bounds.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry) noexcept;

    /**
     * @brief Set the lower world-space bound explicitly.
     *
     * @param v Lower Cartesian bound to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Set the upper world-space bound explicitly.
     *
     * @param v Upper Cartesian bound to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Set the uniform cell size.
     *
     * @param h Cell edge length to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

    /**
     * @brief Set the domain temperature mode.
     *
     * @param type Domain mode to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(DomainType type) noexcept;

    /**
     * @brief Set the prescribed temperature for isothermal domains.
     *
     * @details
     * Stages the temperature value used when the final domain is constructed in
     * @ref DomainType::isothermal mode.
     *
     * @param temperature Prescribed isothermal temperature.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged bounds, discretization,
     * and temperature-mode parameters.
     *
     * @note
     * The exact validation policy is implementation-defined in `domain.hpp`.
     */
    void
    validate() const;

private:
    /**
     * @brief Pending lower world-space bound.
     */
    Vector3<T> _lower_corner { T(0), T(0), T(0) };

    /**
     * @brief Pending upper world-space bound.
     */
    Vector3<T> _upper_corner { T(1), T(1), T(1) };

    /**
     * @brief Pending uniform cell size.
     */
    T _cell_size = T(1);

    /**
     * @brief Pending domain temperature mode.
     */
    DomainType _type = DomainType::variable;

    /**
     * @brief Pending prescribed isothermal temperature.
     *
     * @details
     * Used only when the staged domain mode is @ref DomainType::isothermal.
     */
    std::optional<T> _isothermal_field_temperature;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::DomainType.
 */
using DomainType = atlas::system::DomainType;

/**
 * @brief Convenience alias for @ref atlas::system::Domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Domain = atlas::system::Domain<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::Domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DomainHostPtr = atlas::host_shared_ptr<atlas::system::Domain<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::Domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DomainDevicePtr = atlas::device_shared_ptr<atlas::system::Domain<T>>;

} // namespace atlas

#include <atlas/domain/domain.hpp>