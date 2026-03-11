#pragma once
#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

/**
 * @brief Policy controlling the extent of neighbor cell traversal in spatial searches.
 *
 * @details
 * `NeighborSearchRange` specifies how many surrounding grid cells are visited when
 * performing neighbor queries (e.g., in spatial hashing–based searchers).
 *
 * Typical interpretations:
 * - `single`   → search only the immediate neighborhood (usually the 3×3×3 Moore neighborhood)
 * - `multiple` → search an expanded neighborhood (implementation-defined, e.g., larger stencil)
 *
 * ---
 *
 * @note
 * - The exact semantics of each mode are defined by the concrete `Searcher` implementation.
 * - This enum is intentionally small to keep device-side branching minimal.
 *
 * @see Searcher, SpatialHashingSearcher
 */
enum class NeighborSearchRange : int {
    single,
    multiple
};

/**
 * @brief Abstract base class for particle neighbor search strategies.
 *
 * @details
 * `Searcher<T>` defines a common interface for building acceleration structures
 * used in particle-based neighbor queries (e.g., spatial hashing, uniform grids,
 * BVHs, k-d trees).
 *
 * A `Searcher` consumes particle data (typically positions and active counts)
 * and prepares internal data structures that enable efficient neighbor iteration
 * on the device.
 *
 * Concrete implementations are responsible for:
 * - Allocating and managing internal buffers
 * - Building search structures in `build()`
 * - Providing a device-side probe or equivalent mechanism for neighbor traversal
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @note
 * - This class is intended to be used polymorphically via pointers or references.
 * - `build()` is a **host-entry** function; implementations may launch GPU kernels.
 * - Lifetime management is typically handled via `atlas::host_shared_ptr` /
 *   `atlas::device_shared_ptr`.
 *
 * @see NeighborSearchRange, SpatialHashingSearcher
 */
template <typename T>
class Searcher {
public:
    /// @brief Default constructor.
    Searcher() = default;

    /// @brief Virtual destructor to allow safe polymorphic deletion.
    virtual ~Searcher() = default;

    /**
     * @brief Builds (or rebuilds) the neighbor search data structures.
     *
     * @details
     * Prepares internal acceleration structures based on the provided particle data.
     * Implementations may:
     * - Recompute spatial partitions
     * - Reallocate buffers
     * - Sort or reorganize particle indices
     *
     * This function must be called before any device-side neighbor queries are performed.
     *
     * ---
     *
     * @param particle_probe Device probe describing the current particle state.
     *
     * @note
     * - Expected to be called whenever particle positions or activity states change.
     * - Must be overridden by concrete searcher implementations.
     *
     * @see SpatialHashingSearcher::build()
     */
    virtual void
    build(const system::ParticleDeviceProbe<T>& particle_probe)
        = 0;
};

}

namespace atlas {
/**
 * @brief Alias for a particle neighbor searcher.
 */
template <typename T>
using Searcher = system::Searcher<T>;

/**
 * @brief Host-side shared pointer to a `Searcher<T>`.
 */
template <typename T>
using SearcherHostPtr = atlas::host_shared_ptr<system::Searcher<T>>;

/**
 * @brief Device-side shared pointer to a `Searcher<T>`.
 */
template <typename T>
using SearcherDevicePtr = atlas::device_shared_ptr<system::Searcher<T>>;

} // namespace atlas