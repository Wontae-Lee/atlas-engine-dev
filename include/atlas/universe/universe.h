#pragma once

/**
 * @file universe.h
 * @brief Declares the Universe class representing a regular Cartesian simulation domain and its associated state registry.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/universe/universe_state.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace atlas {

using UniverseStateStore = TypeStore<UniverseState>;

/**
 * @brief Regular Cartesian simulation domain with cell-based state storage.
 *
 * A Universe defines:
 * - the lower and upper corners of the simulation domain,
 * - the grid resolution derived from the cell size,
 * - the number of cells and derived geometric properties,
 * - a registry of universe-side states stored by type.
 *
 * The class acts as the field-side counterpart to particle-based fluid storage.
 * Derived or associated states may store quantities such as temperature, bulk
 * velocity, density-like fields, and other per-cell measurements.
 *
 * @tparam T Floating-point scalar type used by the universe geometry.
 */
template <typename T>
class Universe {
public:
    /**
     * @brief Builder for configuring and constructing Universe objects.
     */
    class Builder;

    /**
     * @brief Default construction is disabled.
     *
     * A Universe must always be created from valid geometric extents and a
     * positive cell size.
     */
    Universe() = delete;

    /**
     * @brief Constructs a universe from domain bounds and cell size.
     *
     * This constructor stores the domain extents, computes derived grid
     * quantities, and initializes the number of cells.
     *
     * @param lower_corner Lower corner of the simulation domain.
     * @param upper_corner Upper corner of the simulation domain.
     * @param cell_size Edge length of a single grid cell.
     */
    Universe(const Vector3<T>& lower_corner,
             const Vector3<T>& upper_corner,
             T cell_size);

    /**
     * @brief Copy construction is disabled.
     */
    Universe(const Universe&) = delete;

    /**
     * @brief Move constructor.
     */
    Universe(Universe&&) noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Universe() = default;

    /**
     * @brief Copy assignment is disabled.
     *
     * @return Reference to this object.
     */
    Universe&
    operator=(const Universe&)
        = delete;

    /**
     * @brief Move assignment operator.
     *
     * @return Reference to this object.
     */
    Universe&
    operator=(Universe&&) noexcept = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent universe construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Constructs and registers a universe state in place.
     *
     * If a state of the same type already exists, it is replaced.
     *
     * @tparam StateT Concrete universe state type.
     * @tparam Args Constructor argument types.
     * @param args Arguments forwarded to the state constructor.
     * @return Reference to the stored state.
     */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args);

    /**
     * @brief Registers a universe state from a unique pointer.
     *
     * If a state of the same type already exists, it is replaced.
     *
     * @tparam StateT Concrete universe state type.
     * @param state Unique pointer to the state to store.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state);

    /**
     * @brief Returns mutable access to a registered state.
     *
     * @tparam StateT Concrete universe state type.
     * @return Pointer to the stored state, or nullptr if not found.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept;

    /**
     * @brief Returns read-only access to a registered state.
     *
     * @tparam StateT Concrete universe state type.
     * @return Const pointer to the stored state, or nullptr if not found.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept;

    /**
     * @brief Returns whether a state of the requested type is registered.
     *
     * @tparam StateT Concrete universe state type.
     * @return True if the state exists, false otherwise.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept;

    /**
     * @brief Removes and returns a registered state.
     *
     * @tparam StateT Concrete universe state type.
     * @return Unique pointer to the removed state, or nullptr if not found.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state();

    /**
     * @brief Returns the total number of grid cells in the universe.
     *
     * @return Number of cells.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    /**
     * @brief Returns the lower corner of the domain.
     *
     * @return Lower corner.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Returns the upper corner of the domain.
     *
     * @return Upper corner.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    /**
     * @brief Returns the grid resolution in each axis.
     *
     * @return Grid size as integer cell counts along x, y, and z.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Returns the cell size.
     *
     * @return Edge length of one cell.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    /**
     * @brief Returns the cell volume.
     *
     * @return Volume of one cell.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    /**
     * @brief Returns the inverse cell size.
     *
     * @return Reciprocal of the cell size.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

    /**
     * @brief Returns the optional observer used to record runtime metrics.
     *
     * @return Const reference to the observer shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const ObserverHostPtr&
    observer() const noexcept;

    /**
     * @brief Returns the full state registry as a mutable reference.
     *
     * @return Mutable reference to the state registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE UniverseStateStore&
    states() noexcept;

    /**
     * @brief Returns the full state registry as a const reference.
     *
     * @return Const reference to the state registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const UniverseStateStore&
    states() const noexcept;

    /**
     * @brief Saves the current universe snapshot to a protobuf-backed binary file.
     *
     * @param path Destination file path.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    save(std::string_view path) const;

private:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<int>
    compute_grid_size(const Vector3<T>& lower_corner,
                      const Vector3<T>& upper_corner,
                      T inverse_cell_size) noexcept;

    /**
     * @brief Lower corner of the domain.
     */
    Vector3<T> _lower_corner;

    /**
     * @brief Upper corner of the domain.
     */
    Vector3<T> _upper_corner;

    /**
     * @brief Grid resolution along each axis.
     */
    Vector3<int> _grid_size { 1, 1, 1 };

    /**
     * @brief Cell edge length.
     */
    T _cell_size = T(1);

    /**
     * @brief Volume of one cell.
     */
    T _cell_volume = T(1);

    /**
     * @brief Reciprocal of the cell size.
     */
    T _inv_h = T(1);

    /**
     * @brief Total number of cells in the grid.
     */
    int _num_of_cells = 1;

    /**
     * @brief Optional observer used to record runtime metrics.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Registry of universe states indexed by concrete type.
     */
    UniverseStateStore _states;
};

/**
 * @brief Builder for Universe.
 *
 * This builder configures the universe from either:
 * - explicit lower/upper corners and cell size, or
 * - a geometry whose bound is used as the universe extent.
 *
 * Validation ensures:
 * - positive cell size,
 * - strictly increasing bounds,
 * - valid computed grid resolution,
 * - no overflow in the total cell count.
 *
 * @tparam T Floating-point scalar type used by the universe geometry.
 */
template <typename T>
class Universe<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a validated Universe object.
     *
     * @return Constructed Universe object.
     *
     * @throw std::invalid_argument Thrown if the configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Universe<T>
    build() const;

    /**
     * @brief Builds a host-side shared Universe object.
     *
     * @return Host shared pointer to a constructed Universe object.
     *
     * @throw std::invalid_argument Thrown if the configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Universe<T>>
    make_host_shared() const;

    /**
     * @brief Sets the universe bounds from a geometry bounding box.
     *
     * @param geometry Host shared pointer to a geometry object.
     * @return Reference to this builder.
     *
     * @throw std::invalid_argument Thrown if geometry is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry);

    /**
     * @brief Sets the lower corner of the universe.
     *
     * @param v Lower corner.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the upper corner of the universe.
     *
     * @param v Upper corner.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    /**
     * @brief Sets the cell size of the universe.
     *
     * @param h Cell size.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

    /**
     * @brief Sets the optional observer used to record runtime metrics.
     *
     * @param observer Host shared pointer to the observer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /**
     * @brief Loads a protobuf-backed universe snapshot into the builder.
     *
     * @param path Snapshot file path.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_binary(const std::string& path);

private:
    /**
     * @brief Validates the current builder configuration.
     *
     * @throw std::invalid_argument Thrown if validation fails.
     */
    void
    validate() const;

private:
    /**
     * @brief Lower corner configured for construction.
     */
    Vector3<T> _lower_corner { T(0), T(0), T(0) };

    /**
     * @brief Upper corner configured for construction.
     */
    Vector3<T> _upper_corner { T(1), T(1), T(1) };

    /**
     * @brief Cell size configured for construction.
     */
    T _cell_size = T(1);

    /**
     * @brief Optional observer installed into the built universe.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Optional restored temperature state.
     */
    std::optional<HostBuffer<T>> _temperature_state;

    /**
     * @brief Optional restored bulk-velocity state.
     */
    std::optional<HostBuffer<Vector3<T>>> _bulk_velocity_state;

    /**
     * @brief Optional restored field-force state.
     */
    std::optional<HostBuffer<Vector3<T>>> _field_force_state;

    /**
     * @brief Optional restored maximum-relative-speed state.
     */
    std::optional<HostBuffer<T>> _max_relative_speed_state;

    /**
     * @brief Optional restored thermal-energy state.
     */
    std::optional<HostBuffer<T>> _thermal_energy_state;

    /**
     * @brief Optional restored number-particle state.
     */
    std::optional<HostBuffer<T>> _number_particle_state;

    /**
     * @brief Optional restored collision-count state.
     */
    std::optional<HostBuffer<int>> _collision_count_state;

    /**
     * @brief Optional restored Knudsen-number state.
     */
    std::optional<HostBuffer<T>> _knudsen_number_state;
};

} // namespace atlas

namespace atlas {


/**
 * @brief Host-side shared pointer alias for Universe.
 *
 * @tparam T Floating-point scalar type used by the universe.
 */
template <typename T>
using UniverseHostPtr = atlas::host_shared_ptr<atlas::Universe<T>>;

/**
 * @brief Device-side shared pointer alias for Universe.
 *
 * @tparam T Floating-point scalar type used by the universe.
 */
template <typename T>
using UniverseDevicePtr = atlas::device_shared_ptr<atlas::Universe<T>>;

} // namespace atlas

#include <atlas/universe/universe.hpp>
