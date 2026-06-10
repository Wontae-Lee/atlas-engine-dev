#pragma once

/**
 * @file octree_searcher.h
 * @brief Declares an octree searcher implementation.
 */

#include <atlas/searcher/searcher.h>

namespace atlas::system {

/**
 * @brief Octree-style particle searcher using 3D domain partitioning.
 *
 * This implementation keeps the common cell-compatible buffers from Searcher<T>
 * and builds neighbor lists by first assigning particles to one of eight root
 * octants. Only particles in the same octant are checked with the exact
 * squared-distance test.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class OctreeSearcher final : public Searcher<T> {
public:
    /**
     * @brief Builder type used for host-side construction.
     */
    class Builder;

    /**
     * @brief Default constructor for deferred initialization.
     */
    OctreeSearcher() = default;

    /**
     * @brief Constructs an octree searcher from required runtime dependencies.
     *
     * @param universe Host pointer supplying domain and grid configuration.
     * @param fluid Host pointer supplying particle state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit
    OctreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    /**
     * @brief Default destructor.
     */
    ~OctreeSearcher() override = default;

    /**
     * @brief Rebuilds the grid-compatible view and octree neighbor list.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void build() override;

    /**
     * @brief Creates a builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder builder() noexcept;

private:
    /**
     * @brief Builds neighbor slots from same-octant candidates.
     *
     * @param alive Number of active particles.
     * @param pos Pointer to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

/**
 * @brief Host-side builder for OctreeSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class OctreeSearcher<T>::Builder final {
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
    ATLAS_HOST ATLAS_FORCE_INLINE Builder& with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @param fluid Host pointer to the fluid object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder& with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Validates dependencies and constructs a searcher value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE OctreeSearcher<T> build() const;

    /**
     * @brief Validates dependencies and constructs a shared host searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<OctreeSearcher<T>> make_host_shared() const;

private:
    /**
     * @brief Validates that required dependencies have been set.
     */
    void validate() const;

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

} // namespace atlas::system

namespace atlas {

template <typename T>
using OctreeSearcher = atlas::system::OctreeSearcher<T>;

/**
 * @brief Host shared pointer alias using the common Searcher interface.
 */
template <typename T>
using OctreeSearcherHostPtr = atlas::host_shared_ptr<atlas::system::Searcher<T>>;

} // namespace atlas

#include <atlas/searcher/octree_searcher.hpp>
