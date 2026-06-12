#pragma once

/**
 * @file kdtree_searcher.h
 * @brief Declares a KD-tree searcher implementation.
 */

#include <atlas/searcher/searcher.h>

namespace atlas {

/**
 * @brief KD-tree style particle searcher using axis prefiltering.
 *
 * This implementation keeps the common cell-compatible buffers from Searcher<T>
 * and builds its neighbor list with a KD-tree inspired candidate reduction:
 * candidates outside the search radius on the x-axis are rejected before the
 * exact squared-distance test is evaluated.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class KdTreeSearcher final : public Searcher<T> {
public:
    /**
     * @brief Builder type used for host-side construction.
     */
    class Builder;

    /**
     * @brief Default constructor for deferred initialization.
     */
    KdTreeSearcher() = default;

    /**
     * @brief Constructs a KD-tree searcher from required runtime dependencies.
     *
     * @param universe Host pointer supplying domain and grid configuration.
     * @param fluid Host pointer supplying particle state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit
    KdTreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    /**
     * @brief Default destructor.
     */
    ~KdTreeSearcher() override = default;

    /**
     * @brief Rebuilds the grid-compatible view and KD-tree neighbor list.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void build() override;

    /**
     * @brief Creates a builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder builder() noexcept;

public:
    /**
     * @brief Builds neighbor slots using x-axis pruning followed by exact distance checks.
     *
     * @param alive Number of active particles.
     * @param pos Pointer to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_neighbors(int alive, const Vector3<T>* pos);
};

/**
 * @brief Host-side builder for KdTreeSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class KdTreeSearcher<T>::Builder final {
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
    ATLAS_HOST ATLAS_FORCE_INLINE KdTreeSearcher<T> build() const;

    /**
     * @brief Validates dependencies and constructs a shared host searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KdTreeSearcher<T>> make_host_shared() const;

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

} // namespace atlas

namespace atlas {
/**
 * @brief Host shared pointer alias using the common Searcher interface.
 */
template <typename T>
using KdTreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher<T>>;

} // namespace atlas

#include <atlas/searcher/kdtree_searcher.hpp>
