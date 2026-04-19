#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief GPU-oriented spatial hashing searcher for particle neighborhood queries.
 *
 * This class builds and stores a spatial hash structure over active fluid particles.
 * It manages device-side buffers for hash keys, sorted particle indices, and
 * per-cell start/end ranges so neighbor queries can later be evaluated efficiently.
 *
 * The searcher depends on:
 * - a universe object that provides the simulation domain and grid configuration
 * - a fluid object that provides particle state such as positions and alive count
 *
 * @tparam T Floating-point scalar type used for geometric and simulation quantities.
 */
template <typename T>
class SpatialHashingSearcher final {
public:
    /**
     * @brief Builder type used for host-side construction.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * Constructs an empty searcher with no bound universe or fluid instance.
     */
    SpatialHashingSearcher() = default;

    /**
     * @brief Constructs a searcher from the required universe and fluid objects.
     *
     * @param universe Host pointer to the universe object.
     * @param fluid Host pointer to the fluid object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SpatialHashingSearcher(
        UniverseHostPtr<T> universe,
        FluidHostPtr<T> fluid);

    /**
     * @brief Default destructor.
     */
    ~SpatialHashingSearcher() = default;

    /**
     * @brief Builds the spatial hashing structure from the current simulation state.
     *
     * This function is expected to prepare buffers, compute particle hash keys,
     * sort active particles by key, and generate cell start/end lookup ranges.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build();

    /**
     * @brief Resets the searcher state and releases or clears internal buffers as needed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Creates a builder instance for host-side construction.
     *
     * @return Builder object initialized for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Ensures internal device buffers are sized for the given number of alive particles.
     *
     * @param alive Number of currently active/alive particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_buffers(int alive);

    /**
     * @brief Initializes the index buffer with an iota sequence.
     *
     * This typically produces the sequence [0, 1, 2, ..., n_active - 1]
     * before reordering by hash key.
     *
     * @param n_active Number of active particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int n_active);

    /**
     * @brief Computes spatial hash keys from particle positions.
     *
     * @param alive Number of active/alive particles.
     * @param pos Pointer to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_keys(int alive, const Vector3<T>* pos);

    /**
     * @brief Sorts particle indices according to their computed hash keys.
     *
     * @param active Number of active particles to sort.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int active);

    /**
     * @brief Builds per-cell start and end index ranges from the sorted key array.
     *
     * These ranges allow fast lookup of particles belonging to each spatial hash cell.
     *
     * @param alive Number of active/alive particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    /**
     * @brief Returns the world-space lower corner of the spatial hash grid.
     *
     * @return Lower corner position of the grid domain.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Returns the grid resolution in each axis.
     *
     * @return Integer grid size along x, y, and z.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Returns the inverse of the cell size.
     *
     * This value is commonly used to map positions into grid coordinates efficiently.
     *
     * @return Inverse cell size.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

    /**
     * @brief Returns the cell size used by the spatial hash grid.
     *
     * @return Cell edge length.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    /**
     * @brief Returns the device pointer to the sorted particle index array.
     *
     * @return Pointer to sorted particle indices.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    indices() const noexcept;

    /**
     * @brief Returns the device pointer to the cell-start lookup array.
     *
     * Each entry stores the first sorted-particle index belonging to a given cell.
     *
     * @return Pointer to cell-start array.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    cell_start() const noexcept;

    /**
     * @brief Returns the device pointer to the cell-end lookup array.
     *
     * Each entry stores the one-past-last sorted-particle index belonging to a given cell.
     *
     * @return Pointer to cell-end array.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    cell_end() const noexcept;

    /**
     * @brief Converts 3D grid coordinates into a linear cell key.
     *
     * @param ix Cell index along the x-axis.
     * @param iy Cell index along the y-axis.
     * @param iz Cell index along the z-axis.
     * @param gs Grid resolution along each axis.
     * @return Linearized cell key.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

private:
    /**
     * @brief Bound universe object providing domain and grid information.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Bound fluid object providing particle data.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Device buffer storing per-particle spatial hash keys.
     */
    DeviceBuffer<std::uint32_t> d_keys;

    /**
     * @brief Device buffer storing particle indices sorted by spatial hash key.
     */
    DeviceBuffer<int> d_indices;

    /**
     * @brief Device buffer storing the inclusive start index for each hash cell.
     */
    DeviceBuffer<int> d_cell_start;

    /**
     * @brief Device buffer storing the exclusive end index for each hash cell.
     */
    DeviceBuffer<int> d_cell_end;
};

/**
 * @brief Host-side builder for SpatialHashingSearcher construction.
 *
 * This builder gathers the required dependencies, validates them, and produces
 * a fully initialized searcher instance.
 *
 * @tparam T Floating-point scalar type used by the target searcher.
 */
template <typename T>
class SpatialHashingSearcher<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @param universe Host pointer to the universe object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @param fluid Host pointer to the fluid object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Validates the current builder state and constructs a searcher instance.
     *
     * @return Fully constructed SpatialHashingSearcher object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingSearcher<T>
    build() const;

    /**
     * @brief Builds a searcher instance and wraps it in a host shared pointer.
     *
     * @return Shared host pointer owning the constructed searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpatialHashingSearcher<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder state before construction.
     */
    void
    validate() const;

private:
    /**
     * @brief Stored universe dependency.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Stored fluid dependency.
     */
    FluidHostPtr<T> _fluid {};
};

}

namespace atlas {

/**
 * @brief Alias for atlas::system::SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;

/**
 * @brief Host shared pointer alias for SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;

/**
 * @brief Device shared pointer alias for SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;

}

#include <atlas/searcher/spatial_hashing_searcher.hpp>