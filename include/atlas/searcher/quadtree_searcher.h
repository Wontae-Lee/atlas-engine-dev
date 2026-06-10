#pragma once

/**
 * @file quadtree_searcher.h
 * @brief Declares a quadtree searcher implementation.
 */

#include <atlas/searcher/searcher.h>

namespace atlas::system {

/**
 * @brief Quadtree-style particle searcher using XY domain partitioning.
 *
 * This implementation keeps the common cell-compatible buffers from Searcher<T>
 * and builds neighbor lists by first assigning particles to one of four root
 * quadrants in the XY plane. Same-quadrant candidates are then filtered by the
 * exact 3D squared-distance test.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class QuadtreeSearcher final : public Searcher<T> {
public:
    /**
     * @brief Builder type used for host-side construction.
     */
    class Builder;

    /**
     * @brief Default constructor for deferred initialization.
     */
    QuadtreeSearcher() = default;

    /**
     * @brief Constructs a quadtree searcher from required runtime dependencies.
     *
     * @param universe Host pointer supplying domain and grid configuration.
     * @param fluid Host pointer supplying particle state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit
    QuadtreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    /**
     * @brief Default destructor.
     */
    ~QuadtreeSearcher() override = default;

    /**
     * @brief Rebuilds the grid-compatible view and quadtree neighbor list.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void build() override;

    /**
     * @brief Creates a builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder builder() noexcept;

public:
    /**
     * @brief Builds neighbor slots from same-quadrant candidates.
     *
     * @param alive Number of active particles.
     * @param pos Pointer to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

/**
 * @brief Host-side builder for QuadtreeSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class QuadtreeSearcher<T>::Builder final {
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
    ATLAS_HOST ATLAS_FORCE_INLINE QuadtreeSearcher<T> build() const;

    /**
     * @brief Validates dependencies and constructs a shared host searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<QuadtreeSearcher<T>> make_host_shared() const;

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
using QuadtreeSearcher = atlas::system::QuadtreeSearcher<T>;

/**
 * @brief Host shared pointer alias using the common Searcher interface.
 */
template <typename T>
using QuadtreeSearcherHostPtr = atlas::host_shared_ptr<atlas::system::Searcher<T>>;

} // namespace atlas

#include <atlas/searcher/quadtree_searcher.hpp>
