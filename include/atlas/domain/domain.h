#pragma once

/**
 * @file domain.h
 * @brief Declares the simulation domain, domain probe, and builder API.
 *
 * The domain owns grid-aligned field storage and geometric metadata used by
 * search, measurement, encoding, and solver stages. It also provides a compact
 * device probe so backend kernels can access field buffers and grid parameters
 * without carrying host-side ownership.
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
 * @brief Selects how domain temperature should be interpreted.
 */
enum class DomainType : int {
    isothermal, ///< The field temperature is fixed and prescribed everywhere.
    variable    ///< The field temperature may evolve during the simulation.
};

/**
 * @brief Device-facing view of domain state.
 *
 * DomainDeviceProbe is the compact structure passed into device kernels and
 * backend-parallel lambdas. It exposes raw field pointers and precomputed grid
 * metadata required for cell lookup and field updates.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct DomainDeviceProbe {

    DomainType type = DomainType::variable; ///< Runtime domain temperature mode.

    T* field_temperature = nullptr; ///< Per-cell temperature field.

    Vector3<T>* field_force = nullptr; ///< Optional per-cell external force field.

    Vector3<T> lower_corner; ///< World-space minimum corner of the domain bounds.

    Vector3<T> upper_corner; ///< World-space maximum corner of the domain bounds.

    Vector3<int> grid_size { 1, 1, 1 }; ///< Number of cells along each axis.

    T cell_size = T(1); ///< Uniform grid spacing.

    T cell_volume = T(1); ///< Precomputed cubic cell volume.

    T inv_h = T(1); ///< Reciprocal cell size for fast coordinate-to-cell mapping.

    int num_of_cells = 0; ///< Total number of cells in the discretized domain.
};

/**
 * @brief Owns the simulation grid, field storage, and domain metadata.
 *
 * Domain converts geometric bounds and a cell size into a regular Cartesian
 * grid. It stores device-resident temperature and optional force fields, and
 * provides a single authoritative device probe consumed by the system runtime.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Domain {
public:
    /**
     * @brief Fluent builder used to configure and construct Domain instances.
     */
    class Builder;

    /**
     * @brief Deleted default constructor.
     */
    Domain() = delete;

    /**
     * @brief Constructs a domain from explicit bounds and discretization.
     *
     * @param lower_corner Minimum world-space bound.
     * @param upper_corner Maximum world-space bound.
     * @param cell_size Uniform cubic cell spacing.
     * @param type Domain temperature mode.
     * @param temperature Optional prescribed temperature for isothermal domains.
     */
    Domain(const Vector3<T>& lower_corner,
           const Vector3<T>& upper_corner,
           T cell_size,
           DomainType type              = DomainType::variable,
           std::optional<T> temperature = std::nullopt);

    /**
     * @brief Destructor.
     */
    ~Domain() = default;

    /**
     * @brief Returns a builder initialized with default values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Creates the device probe consumed by runtime kernels.
     *
     * The system owns the single authoritative probe for a live simulation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Returns the total number of grid cells.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    /**
     * @brief Returns the configured domain mode.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DomainType
    type() const noexcept;

    /**
     * @brief Returns the prescribed field temperature for isothermal domains.
     *
     * @throws std::runtime_error if the domain is not isothermal or the
     *         temperature was not configured.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    isothermal_field_temperature() const;

    /**
     * @brief Replaces the optional field-force buffer.
     *
     * @param field_force Device buffer storing one force vector per cell.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_field_force(DeviceBuffer<Vector3<T>> field_force) noexcept;

    /**
     * @brief Returns mutable access to the field-force buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    field_force() noexcept;

    /**
     * @brief Returns const access to the field-force buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Vector3<T>>&
    field_force() const noexcept;

    /**
     * @brief Returns the lower world-space bound of the domain.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Returns the upper world-space bound of the domain.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    /**
     * @brief Returns the integer grid resolution.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Returns the uniform cell size.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    /**
     * @brief Returns the volume of a single cubic cell.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    /**
     * @brief Returns the reciprocal of the uniform cell size.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

private:
    DeviceBuffer<T> d_field_temperature; ///< Per-cell temperature field.

    DeviceBuffer<Vector3<T>> d_field_force; ///< Optional per-cell force field.

    Vector3<T> _lower_corner; ///< Minimum world-space domain bound.

    Vector3<T> _upper_corner; ///< Maximum world-space domain bound.

    Vector3<int> _grid_size { 1, 1, 1 }; ///< Cells along x, y, and z.

    T _cell_size = T(1); ///< Uniform grid spacing.

    T _cell_volume = T(1); ///< Cached cell volume.

    T _inv_h = T(1); ///< Cached reciprocal cell size.

    int _num_of_cells = 1; ///< Cached total number of cells.

    DomainType _type = DomainType::variable; ///< Domain temperature mode.

    std::optional<T> _isothermal_field_temperature; ///< Prescribed field temperature for isothermal mode.

    std::uint64_t _probe_count = 0; ///< Tracks runtime probe creation.
};

template <typename T>
class Domain<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a value instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Domain<T>
    build() const;

    /**
     * @brief Builds a host-shared domain instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Domain<T>>
    make_host_shared() const;

    /**
     * @brief Derives the domain bounds from a geometry object's bounding box.
     *
     * @param geometry Geometry providing the source bounds.
     * @return Builder& Fluent reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry) noexcept;

    /**
     * @brief Sets the lower world-space bound explicitly.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the upper world-space bound explicitly.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the uniform cell size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

    /**
     * @brief Sets the domain temperature mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(DomainType type) noexcept;

    /**
     * @brief Sets the prescribed temperature for isothermal domains.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    /**
     * @brief Validates builder state before construction.
     */
    void
    validate() const;

private:
    Vector3<T> _lower_corner { T(0), T(0), T(0) }; ///< Pending lower bound.

    Vector3<T> _upper_corner { T(1), T(1), T(1) }; ///< Pending upper bound.

    T _cell_size = T(1); ///< Pending uniform cell size.

    DomainType _type = DomainType::variable; ///< Pending domain mode.

    std::optional<T> _isothermal_field_temperature; ///< Pending isothermal temperature.
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::DomainType.
 */
using DomainType = atlas::system::DomainType;

/**
 * @brief Convenience alias for atlas::system::Domain.
 */
template <typename T>
using Domain = atlas::system::Domain<T>;
/**
 * @brief Host shared pointer alias for Domain.
 */
template <typename T>
using DomainHostPtr = atlas::host_shared_ptr<atlas::system::Domain<T>>;
/**
 * @brief Device shared pointer alias for Domain.
 */
template <typename T>
using DomainDevicePtr = atlas::device_shared_ptr<atlas::system::Domain<T>>;
}

#include <atlas/domain/domain.hpp>
