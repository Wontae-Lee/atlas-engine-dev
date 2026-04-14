#pragma once

/**
 * @file spatial_hashing_searcher.h
 * @brief Declares the spatial hashing search structure, its device-facing probe, and builder utilities.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::SpatialHashingProbe, a compact device-facing view of
 *   spatial hashing data,
 * - @ref atlas::system::SpatialHashingSearcher, the host-side runtime object
 *   responsible for building and maintaining the spatial hashing structure.
 *
 * ## Purpose
 * Spatial hashing is used to accelerate neighborhood queries over particle data
 * by mapping particle positions into a regular Cartesian grid derived from the
 * simulation domain.
 *
 * This allows runtime systems such as:
 * - solvers,
 * - codecs,
 * - measurement stages,
 * - collision and interaction routines,
 * to efficiently iterate over particles located in nearby cells rather than
 * scanning the full particle set.
 *
 * ## Data model
 * The search structure is built from:
 * - the domain lower corner,
 * - the grid resolution,
 * - the uniform cell size,
 * - a particle position buffer.
 *
 * Runtime query data is stored in compact arrays such as:
 * - sorted particle indices,
 * - cell start offsets,
 * - cell end offsets,
 * - optional intermediate cell keys.
 *
 * ## Probe model
 * Since backend kernels should not carry host-side ownership state, the searcher
 * can export a lightweight @ref SpatialHashingProbe containing:
 * - grid metadata,
 * - raw pointers to the index and cell-range arrays.
 *
 * The system runtime typically owns the single authoritative probe for the live
 * search structure and reuses it across multiple stages.
 *
 * ## Construction
 * A spatial hashing searcher may be:
 * - default-constructed,
 * - directly constructed from a domain,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for spatial coordinates and grid metrics.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Lightweight device-facing view of spatial hashing state.
 *
 * @details
 * @ref SpatialHashingProbe is the compact runtime structure passed to host/device
 * kernels and backend-parallel code that need neighborhood access.
 *
 * It contains:
 * - domain/grid metadata needed to map positions to cells,
 * - raw pointers to the cell-indexing arrays that describe which particles fall
 *   into which cells.
 *
 * ## Ownership
 * All pointer fields are non-owning. The arrays they reference must remain valid
 * for the duration of any backend work that uses the probe.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
struct SpatialHashingProbe {
    /**
     * @brief World-space minimum corner of the hashed domain.
     *
     * @details
     * Used as the origin for coordinate-to-cell mapping.
     */
    Vector3<T> lower_corner {};

    /**
     * @brief Number of cells along each Cartesian axis.
     *
     * @details
     * Stores the regular grid resolution in x, y, and z.
     */
    Vector3<int> grid_size { 0, 0, 0 };

    /**
     * @brief Reciprocal of the cell size.
     *
     * @details
     * Cached for fast coordinate-to-cell mapping.
     */
    T inv_h = T(1);

    /**
     * @brief Uniform cell size.
     *
     * @details
     * Edge length of each spatial hashing grid cell.
     */
    T cell_size = T(1);

    /**
     * @brief Pointer to the sorted particle-index array.
     *
     * @details
     * Each entry maps a sorted spatial-hash position to an original particle index.
     */
    const int* indices {};

    /**
     * @brief Pointer to the cell-start array.
     *
     * @details
     * For each linear cell index, stores the first position in @ref indices that
     * belongs to that cell, or an implementation-defined sentinel when empty.
     */
    const int* cell_start {};

    /**
     * @brief Pointer to the cell-end array.
     *
     * @details
     * For each linear cell index, stores one-past-the-end position in @ref indices
     * for that cell, or an implementation-defined sentinel when empty.
     */
    const int* cell_end {};
};

/**
 * @brief Host-side spatial hashing search structure for neighborhood queries.
 *
 * @details
 * @ref SpatialHashingSearcher builds and owns the runtime data needed to perform
 * grid-based neighborhood lookup over particle positions.
 *
 * It derives its spatial discretization from the associated @ref Domain and
 * stores device-resident arrays for:
 * - hashed particle keys,
 * - sorted particle indices,
 * - per-cell start offsets,
 * - per-cell end offsets.
 *
 * ## Build process
 * A typical build/update sequence consists of:
 * 1. preparing buffers sized to the current active particle count,
 * 2. initializing particle indices,
 * 3. computing spatial hash keys from particle positions,
 * 4. sorting particles by key,
 * 5. building cell start/end ranges for neighborhood traversal.
 *
 * This sequence is exposed both as a single @ref build entry point and as finer-
 * grained helper functions.
 *
 * ## Probe export
 * The searcher can create a @ref SpatialHashingProbe through
 * @ref make_device_probe. The system runtime usually caches that probe as the
 * single authoritative device-facing spatial-search view.
 *
 * ## Ownership model
 * The searcher stores:
 * - the associated domain,
 * - the device buffers that back the runtime search structure,
 * - cached raw pointers to those buffers for fast probe creation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class SpatialHashingSearcher final {
public:
    /**
     * @brief Fluent builder for configuring and constructing
     *        @ref SpatialHashingSearcher.
     *
     * @details
     * The builder stages the associated domain, validates it, and constructs either:
     * - a searcher by value, or
     * - a host-owned shared pointer to a searcher.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty searcher with no associated domain.
     */
    SpatialHashingSearcher() = default;

    /**
     * @brief Construct a searcher from a domain.
     *
     * @param domain Associated simulation domain.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SpatialHashingSearcher(
        DomainHostPtr<T> domain);

    /**
     * @brief Default destructor.
     */
    ~SpatialHashingSearcher() = default;

    /**
     * @brief Build or rebuild the spatial hashing structure for the current particle state.
     *
     * @details
     * Consumes the supplied fluid device probe, derives the active particle count
     * and particle positions, and rebuilds the spatial search arrays needed for
     * neighborhood queries.
     *
     * @param particle_probe Device-facing fluid probe describing active particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const system::FluidDeviceProbe<T>& particle_probe);

    /**
     * @brief Create the device probe consumed by runtime kernels.
     *
     * @details
     * Returns a compact @ref SpatialHashingProbe containing raw pointers and
     * grid metadata derived from this searcher.
     *
     * @return Device-facing spatial hashing probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Reset internal runtime state.
     *
     * @details
     * Reinitializes internal buffers or bookkeeping according to the implementation
     * policy in `spatial_hashing_searcher.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Prepare internal buffers for a given number of active particles.
     *
     * @param alive Number of active/alive particles to accommodate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_buffers(int alive);

    /**
     * @brief Initialize the particle-index array with an iota sequence.
     *
     * @details
     * Typically initializes:
     * - `indices[i] = i`
     * for the active particle prefix.
     *
     * @param n_active Number of active particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int n_active);

    /**
     * @brief Compute spatial hash keys from particle positions.
     *
     * @param alive Number of active/alive particles.
     * @param pos Pointer to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_keys(int alive, const Vector3<T>* pos);

    /**
     * @brief Sort particle indices by their computed spatial hash keys.
     *
     * @param active Number of active particles to sort.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int active) const;

    /**
     * @brief Build per-cell start and end offsets from the sorted key sequence.
     *
     * @param alive Number of active/alive particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    /**
     * @brief Convert integer cell coordinates to a linear hash key.
     *
     * @details
     * The exact linearization order is implementation-defined but is expected to
     * be consistent with the search buffers and probe logic.
     *
     * @param ix Cell coordinate along x.
     * @param iy Cell coordinate along y.
     * @param iz Cell coordinate along z.
     * @param gs Grid resolution.
     * @return Linearized hash key.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

private:
    /**
     * @brief Internal counter tracking probe creation.
     */
    std::uint64_t _probe_count = 0;

    /**
     * @brief Associated simulation domain.
     *
     * @details
     * Supplies the grid bounds and resolution used by the searcher.
     */
    DomainHostPtr<T> _domain {};

    /**
     * @brief Device-resident spatial hash keys.
     *
     * @details
     * Stores one hash key per active particle before and/or after sorting.
     */
    DeviceBuffer<std::uint32_t> d_keys;

    /**
     * @brief Device-resident sorted particle indices.
     *
     * @details
     * Maps sorted hash entries back to original particle indices.
     */
    DeviceBuffer<int> d_indices;

    /**
     * @brief Device-resident per-cell start offsets.
     */
    DeviceBuffer<int> d_cell_start;

    /**
     * @brief Device-resident per-cell end offsets.
     */
    DeviceBuffer<int> d_cell_end;

    /**
     * @brief Cached raw pointer to the hash-key buffer.
     */
    std::uint32_t* d_keys_ptr = nullptr;

    /**
     * @brief Cached raw pointer to the sorted index buffer.
     */
    int* d_indices_ptr = nullptr;

    /**
     * @brief Cached raw pointer to the cell-start buffer.
     */
    int* d_cell_start_ptr = nullptr;

    /**
     * @brief Cached raw pointer to the cell-end buffer.
     */
    int* d_cell_end_ptr = nullptr;
};

/**
 * @brief Fluent builder for @ref SpatialHashingSearcher.
 *
 * @details
 * The builder provides a controlled construction path for the spatial hashing
 * searcher by staging the associated domain.
 *
 * ## Typical usage
 * @code
 * auto searcher = atlas::SpatialHashingSearcher<float>::builder()
 *     .with_domain(domain)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - a valid domain is present,
 * - the domain grid metadata is usable for hashing.
 *
 * The exact validation rules are implementation-defined in
 * `spatial_hashing_searcher.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class SpatialHashingSearcher<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no associated domain.
     */
    Builder() = default;

    /**
     * @brief Set the associated domain.
     *
     * @param domain Domain used to derive spatial hashing grid metadata.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Build a configured @ref SpatialHashingSearcher by value after validation.
     *
     * @return Constructed searcher value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingSearcher<T>
    build() const;

    /**
     * @brief Build a configured @ref SpatialHashingSearcher in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<SpatialHashingSearcher<T>>` owning the constructed searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpatialHashingSearcher<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged domain.
     */
    void
    validate() const;

private:
    /**
     * @brief Pending associated domain.
     */
    DomainHostPtr<T> _domain {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;

} // namespace atlas

#include <atlas/searcher/spatial_hashing_searcher.hpp>
