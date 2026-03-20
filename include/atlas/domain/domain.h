#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

/**
 * @brief Lightweight device-side view of a simulation domain.
 *
 * @details
 * `DomainDeviceProbe` is a plain-data structure intended to be copied to and used on the GPU.
 * It provides:
 * - Raw device pointers to per-cell buffers (e.g., `temperature`, `field_force`)
 * - Geometric bounds of the domain (`lower_corner`, `upper_corner`)
 * - Discretization parameters (`grid_size`, `cell_size`, `cell_volume`, `inv_h`)
 * - Total number of cells (`num_of_cells`)
 *
 * The domain is assumed to be discretized into a regular 3D grid of axis-aligned cubic cells
 * with uniform spacing `cell_size` (a.k.a. \f$h\f$). The inverse spacing is stored as
 * \f$\text{inv\_h} = 1/h\f$ for fast index/coordinate conversions.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @note
 * - This structure intentionally avoids owning memory; it only references device buffers.
 * - The producer (`Domain::make_device_probe()`) must ensure pointers remain valid for the
 *   lifetime of GPU kernels using the probe.
 * - `grid_size` is the number of cells along each axis; `num_of_cells` is typically
 *   `grid_size.x * grid_size.y * grid_size.z`.
 *
 * @see Domain, Domain::make_device_probe()
 */
template <typename T>
struct DomainDeviceProbe {
    /// @brief Device pointer to per-cell temperature values (size = `number_of_cells`).
    T* temperature = nullptr;

    /// @brief Device pointer to per-cell force vectors (size = `number_of_cells`).
    Vector3<T>* field_force = nullptr;

    /// @brief Lower (minimum) corner of the domain in world coordinates.
    Vector3<T> lower_corner;

    /// @brief Upper (maximum) corner of the domain in world coordinates.
    Vector3<T> upper_corner;

    /// @brief Number of grid cells along each axis (X, Y, Z).
    Vector3<int> grid_size { 1, 1, 1 };

    /// @brief Uniform cell edge length \f$h\f$.
    T cell_size = T(1);

    /// @brief Cell volume \f$h^3\f$ for cubic cells.
    T cell_volume = T(1);

    /// @brief Inverse cell size \f$1/h\f$ (precomputed for performance).
    T inv_h = T(1);

    /// @brief Total number of cells in the grid.
    int num_of_cells = 0;
};

/**
 * @brief Uniform 3D simulation domain discretized into a regular grid of cubic cells.
 *
 * @details
 * `Domain` represents an axis-aligned 3D region \f$[\mathbf{l}, \mathbf{u}]\f$ with:
 * - World-space bounds (`_lower_corner`, `_upper_corner`)
 * - A uniform cell size \f$h\f$ (`_cell_size`)
 * - A derived grid resolution (`_grid_size`) and total cell count (`_num_of_cells`)
 *
 * It owns device buffers storing per-cell quantities, and can produce a lightweight
 * `DomainDeviceProbe` containing raw device pointers and cached parameters for kernels.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @note
 * - Construction typically computes:
 *   - `cell_volume = h^3`
 *   - `inv_h = 1/h`
 *   - `grid_size = floor((upper - lower) / h)` (implementation-defined rounding policy)
 *   - `num_of_cells = grid_size.x * grid_size.y * grid_size.z`
 * - The class is non-default-constructible to enforce a fully specified domain.
 *
 * @see DomainDeviceProbe, Domain::Builder, Domain::make_device_probe()
 */
template <typename T>
class Domain {
public:
    /**
     * @brief Fluent builder for constructing `Domain` instances.
     *
     * @details
     * The builder collects parameters (`lower_corner`, `upper_corner`, `cell_size`),
     * validates them, and then constructs the domain (and related derived quantities).
     *
     * @see Domain::builder()
     */
    class Builder;

    /// @brief Deleted default constructor — a domain must be fully specified.
    Domain() = delete;

    /**
     * @brief Constructs a domain from bounds and uniform cell size.
     *
     * @param lower_corner Lower (minimum) world-space corner \f$\mathbf{l}\f$.
     * @param upper_corner Upper (maximum) world-space corner \f$\mathbf{u}\f$.
     * @param cell_size    Uniform cell edge length \f$h\f$.
     *
     * @note
     * - It is expected that `upper_corner >= lower_corner` component-wise.
     * - `cell_size` must be strictly positive.
     */
    Domain(const Vector3<T>& lower_corner,
           const Vector3<T>& upper_corner,
           T cell_size);

    /// @brief Default destructor.
    ~Domain() = default;

    /**
     * @brief Returns a builder object for fluent domain construction.
     *
     * @return Domain::Builder instance.
     *
     * @note
     * Host-only because builders are a construction-time convenience.
     *
     * @see Domain::Builder
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Creates a device-side probe containing raw pointers and cached parameters.
     *
     * @details
     * Produces a `DomainDeviceProbe<T>` that references the internal device buffers.
     * The probe can be passed by value to GPU kernels to avoid repeated accessors
     * and to keep hot parameters (e.g., `inv_h`, `grid_size`) readily available.
     *
     * @return DomainDeviceProbe\<T\> with valid device pointers and domain metadata.
     *
     * @note
     * - The returned probe does not own memory.
     * - The pointers remain valid as long as the `Domain` instance (and its buffers)
     *   remain alive and are not reallocated.
     *
     * @see DomainDeviceProbe
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Returns the number of cells.
     *
     * @return Total number of cells in the grid.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    /**
     * @brief Get the domain's lower (minimum) corner in world coordinates.
     *
     * @details
     * Returns the minimum corner of the simulation/domain AABB.
     * This is the point \f$\mathbf{l}=(l_x,l_y,l_z)\f$ such that all valid domain
     * positions satisfy \f$\mathbf{p}\ge\mathbf{l}\f$ component-wise.
     *
     * @tparam T Floating-point scalar type.
     * @return Vector3<T> Lower corner \f$\mathbf{l}\f$.
     *
     * @note
     * - This accessor is host-only (`ATLAS_HOST`).
     * - The returned value is a copy (no reference lifetime concerns).
     *
     * @see upper_corner(), grid_size()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Get the domain's upper (maximum) corner in world coordinates.
     *
     * @details
     * Returns the maximum corner of the simulation/domain AABB.
     * This is the point \f$\mathbf{u}=(u_x,u_y,u_z)\f$ such that all valid domain
     * positions satisfy \f$\mathbf{p}\le\mathbf{u}\f$ component-wise.
     *
     * @tparam T Floating-point scalar type.
     * @return Vector3<T> Upper corner \f$\mathbf{u}\f$.
     *
     * @note
     * - This accessor is host-only (`ATLAS_HOST`).
     * - The returned value is a copy.
     *
     * @see lower_corner(), grid_size()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    /**
     * @brief Get the integer grid resolution of the domain.
     *
     * @details
     * Returns the number of cells along each axis: \f$\mathbf{g}=(g_x,g_y,g_z)\f$.
     * Typically it is derived from the domain extents and `cell_size()`:
     * \f[
     *   g_i \approx \left\lceil \frac{u_i - l_i}{h} \right\rceil,\quad i\in\{x,y,z\}
     * \f]
     * where \f$h=\text{cell\_size}\f$.
     *
     * @return Vector3<int> Grid size (cells per axis).
     *
     * @note
     * - Values are expected to be \f$\ge 1\f$ per axis for a valid domain.
     * - This accessor is host-only (`ATLAS_HOST`).
     *
     * @see cell_size(), cell_volume(), inv_h()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Get the uniform cell size \f$h\f$ used by the domain grid.
     *
     * @details
     * The domain is discretized into a regular Cartesian grid with cubic cells.
     * `cell_size()` returns the edge length \f$h\f$ of each cell in world units.
     *
     * @return T Cell edge length \f$h\f$.
     *
     * @note
     * - Must be positive for a valid domain.
     * - Used by spatial hashing / neighbor search to map positions to cell indices.
     *
     * @see inv_h(), cell_volume(), grid_size()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    /**
     * @brief Get the volume of a single grid cell.
     *
     * @details
     * For a cubic cell with edge length \f$h\f$, the volume is:
     * \f[
     *   V = h^3
     * \f]
     *
     * @return T Cell volume \f$V\f$.
     *
     * @note
     * - This is typically precomputed and cached to avoid repeated `pow` calls.
     * - Assumes uniform cubic cells.
     *
     * @see cell_size(), inv_h()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    /**
     * @brief Get the reciprocal cell size \f$1/h\f$.
     *
     * @details
     * Returns \f$h^{-1}\f$ where \f$h=\text{cell\_size()}\f$:
     * \f[
     *   \text{inv\_h} = \frac{1}{h}
     * \f]
     *
     * This value is commonly used to convert world coordinates to grid-space
     * indices efficiently:
     * \f[
     *   i = \left\lfloor (p_x - l_x)\,\text{inv\_h} \right\rfloor
     * \f]
     * (similarly for \f$y,z\f$).
     *
     * @return T Reciprocal of the cell size.
     *
     * @note
     * - `cell_size()` must be non-zero to keep this finite.
     * - Cached for performance in spatial hashing and uniform-grid operations.
     *
     * @see cell_size(), grid_size()
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

private:
    /// @brief Device buffer storing per-cell temperature values.
    DeviceBuffer<T> d_temperature;

    /// @brief Device buffer storing per-cell force vectors.
    DeviceBuffer<Vector3<T>> d_field_force;

    /// @brief Lower (minimum) world-space corner \f$\mathbf{l}\f$.
    Vector3<T> _lower_corner;

    /// @brief Upper (maximum) world-space corner \f$\mathbf{u}\f$.
    Vector3<T> _upper_corner;

    /// @brief Number of cells along each axis (X, Y, Z).
    Vector3<int> _grid_size { 1, 1, 1 };

    /// @brief Uniform cell edge length \f$h\f$.
    T _cell_size = T(1);

    /// @brief Cell volume \f$h^3\f$.
    T _cell_volume = T(1);

    /// @brief Inverse cell size \f$1/h\f$.
    T _inv_h = T(1);

    /// @brief Total number of cells in the grid.
    int _num_of_cells = 1;

    /// @brief Total number of device probes created.
    std::uint64_t _probe_count = 0;
};

/**
 * @brief Fluent builder for `Domain<T>`.
 *
 * @details
 * Collects parameters required to build a `Domain<T>` and validates them prior to construction.
 * Typical usage:
 * @code
 * auto domain = Domain<float>::builder()
 *                 .with_lower_corner({0,0,0})
 *                 .with_upper_corner({1,1,1})
 *                 .with_cell_size(0.01f)
 *                 .build();
 * @endcode
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @note
 * - `validate()` is expected to enforce:
 *   - `cell_size > 0`
 *   - `upper_corner >= lower_corner` component-wise (and potentially non-degenerate policy)
 *
 * @see Domain, Domain::builder()
 */
template <typename T>
class Domain<T>::Builder final {
public:
    /// @brief Default constructor.
    Builder() = default;

    /**
     * @brief Builds a `Domain<T>` from the current builder state.
     *
     * @details
     * Validates parameters and returns a fully constructed `Domain<T>`.
     *
     * @return Constructed `Domain<T>` instance.
     *
     * @throws std::runtime_error (or project-specific exception) if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Domain<T>
    build() const;

    /**
     * @brief Builds a `Domain<T>` and returns it as a host shared pointer.
     *
     * @details
     * Convenience helper for owning the domain via `atlas::host_shared_ptr`.
     *
     * @return `atlas::host_shared_ptr<Domain<T>>` owning the constructed domain.
     *
     * @throws std::runtime_error (or project-specific exception) if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Domain<T>>
    make_host_shared() const;

    /** @brief Configures the builder's bounds from a geometry object.
     *
     * @param geometry Geometry object providing domain bounds via its query operator.
     * @return Reference to this builder for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry) noexcept;

    /**
     * @brief Sets the lower (minimum) corner of the domain.
     *
     * @param v Lower corner \f$\mathbf{l}\f$ in world coordinates.
     * @return Reference to this builder for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the upper (maximum) corner of the domain.
     *
     * @param v Upper corner \f$\mathbf{u}\f$ in world coordinates.
     * @return Reference to this builder for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the uniform cell size \f$h\f$.
     *
     * @param h Cell edge length (must be > 0).
     * @return Reference to this builder for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

private:
    /**
     * @brief Validates builder parameters or throws on error.
     *
     * @details
     * Expected checks include:
     * - `cell_size` is strictly positive
     * - `upper_corner` is not less than `lower_corner` component-wise
     *
     * @throws std::runtime_error (or project-specific exception) if invalid.
     */
    void
    validate() const;

private:
    /// @brief Cached lower corner.
    Vector3<T> _lower_corner { T(0), T(0), T(0) };

    /// @brief Cached upper corner.
    Vector3<T> _upper_corner { T(1), T(1), T(1) };

    /// @brief Cached cell size.
    T _cell_size = T(1);
};

} // namespace atlas::system

namespace atlas {
template <typename T>
using Domain = atlas::system::Domain<T>;
template <typename T>
using DomainHostPtr = atlas::host_shared_ptr<atlas::system::Domain<T>>;
template <typename T>
using DomainDevicePtr = atlas::device_shared_ptr<atlas::system::Domain<T>>;
}

#include <atlas/domain/domain.hpp>
