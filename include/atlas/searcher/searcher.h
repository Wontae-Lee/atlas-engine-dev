#pragma once

/**
 * @file searcher.h
 * @brief Declares the common particle searcher interface and shared storage.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

#include <cstdint>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Common base for particle search structures.
 *
 * The base class owns the shared dependencies and output buffers used by solver
 * probes. Concrete implementations choose how to build neighbor information,
 * while preserving the cell-compatible view required by existing runtime code.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Searcher {
    static_assert(std::is_floating_point_v<T>, "Searcher requires a floating-point T");

public:
    /**
     * @brief Default constructor for containers and deferred dependency binding.
     *
     * A default-constructed searcher does not have valid universe or fluid
     * dependencies. Concrete searchers should be built through their builders
     * before participating in a simulation step.
     */
    Searcher() = default;

    /**
     * @brief Constructs the common searcher state from required runtime objects.
     *
     * @param universe Host pointer supplying domain bounds, grid resolution, and cell size.
     * @param fluid Host pointer supplying active particle count and particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Searcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    /**
     * @brief Virtual destructor for polymorphic ownership through SearcherHostPtr.
     */
    virtual ~Searcher() = default;

    Searcher(const Searcher&) = default;
    Searcher& operator=(const Searcher&) = default;
    Searcher(Searcher&&) noexcept = default;
    Searcher& operator=(Searcher&&) noexcept = default;

    /**
     * @brief Rebuilds the search structure when the implementation is invalidated.
     *
     * Concrete implementations must refresh both the grid-compatible cell view
     * and their own particle-neighbor list before clearing the invalidation flag.
     */
    ATLAS_HOST virtual void
    build() = 0;

    /**
     * @brief Marks cached search data as stale.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    invalidate() noexcept;

    /**
     * @brief Clears particle buffers and recreates empty cell range buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset() noexcept;

    /**
     * @brief Returns the lower corner of the bound universe domain.
     *
     * @return Domain lower corner, or zero vector if no universe is bound.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual Vector3<T>
    lower_corner() const noexcept;

    /**
     * @brief Returns the grid resolution used for the cell-compatible search view.
     *
     * @return Grid size, or zero grid if no universe is bound.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual Vector3<int>
    grid_size() const noexcept;

    /**
     * @brief Returns the inverse cell size of the bound universe.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual T
    inverse_cell_size() const noexcept;

    /**
     * @brief Returns the cell size used as the default neighbor search radius.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual T
    cell_size() const noexcept;

    /**
     * @brief Returns sorted active particle indices for the cell-compatible view.
     *
     * The returned pointer is valid until the next build/reset that resizes the
     * internal index buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    indices() const noexcept;

    /**
     * @brief Returns per-cell inclusive start offsets into indices().
     *
     * Empty cells contain -1. Non-empty cells store the first sorted index for
     * the corresponding linear cell key.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    cell_start() const noexcept;

    /**
     * @brief Returns per-cell exclusive end offsets into indices().
     *
     * Empty cells contain -1. Non-empty cells store one past the last sorted
     * index for the corresponding linear cell key.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    cell_end() const noexcept;

    /**
     * @brief Returns per-particle offsets into neighbor_indices().
     *
     * Offsets follow a CSR-like convention: particle i owns the range
     * [neighbor_offsets()[i], neighbor_offsets()[i + 1]). Invalid neighbor slots
     * are stored as -1 inside that range.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    neighbor_offsets() const noexcept;

    /**
     * @brief Returns flattened particle-neighbor indices.
     *
     * Each entry is either an active particle index or -1 when the algorithm's
     * fixed output slot did not accept a candidate.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    neighbor_indices() const noexcept;

    /**
     * @brief Returns the total number of slots in neighbor_indices().
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual int
    neighbor_count() const noexcept;

    /**
     * @brief Converts integer grid coordinates into a linear cell key.
     *
     * @param ix Cell index along x.
     * @param iy Cell index along y.
     * @param iz Cell index along z.
     * @param gs Grid size.
     * @return Linearized cell key.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

protected:
    /**
     * @brief Validates that required runtime dependencies are available.
     *
     * @param owner Name used in validation error messages.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate_dependencies(const char* owner) const;

    /**
     * @brief Returns a raw pointer to active particle positions when available.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const Vector3<T>*
    position_ptr() const noexcept;

    /**
     * @brief Returns the number of active particles from the bound fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE int
    active_count() const noexcept;

    /**
     * @brief Sizes key, index, and cell range buffers for a rebuild.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_grid_buffers(int alive);

    /**
     * @brief Initializes _indices with [0, alive).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int alive);

    /**
     * @brief Computes one clamped linear grid key per active particle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_grid_keys(int alive, const Vector3<T>* positions);

    /**
     * @brief Sorts particle indices by their computed grid keys.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int alive);

    /**
     * @brief Builds per-cell start/end ranges after sorting by key.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    /**
     * @brief Clears the neighbor list buffers and resets the slot count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_neighbors();

protected:
    /**
     * @brief Bound universe dependency shared by all searcher implementations.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Bound fluid dependency shared by all searcher implementations.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Per-particle grid keys used by the cell-compatible view.
     */
    DeviceBuffer<std::uint32_t> _keys;

    /**
     * @brief Active particle indices sorted by _keys.
     */
    DeviceBuffer<int> _indices;

    /**
     * @brief Per-cell first sorted-index lookup, with -1 for empty cells.
     */
    DeviceBuffer<int> _cell_start;

    /**
     * @brief Per-cell one-past-last sorted-index lookup, with -1 for empty cells.
     */
    DeviceBuffer<int> _cell_end;

    /**
     * @brief CSR-like offsets into _neighbor_indices.
     */
    DeviceBuffer<int> _neighbor_offsets;

    /**
     * @brief Flattened particle-neighbor slots.
     */
    DeviceBuffer<int> _neighbor_indices;

    /**
     * @brief Total number of entries stored in _neighbor_indices.
     */
    int _neighbor_count {};

    /**
     * @brief Tracks whether build() must refresh cached search data.
     */
    bool _is_invalidated { true };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using Searcher = atlas::system::Searcher<T>;

template <typename T>
using SearcherHostPtr = atlas::host_shared_ptr<Searcher<T>>;

template <typename T>
using SearcherDevicePtr = atlas::device_shared_ptr<Searcher<T>>;

} // namespace atlas

#include <atlas/searcher/searcher.hpp>
