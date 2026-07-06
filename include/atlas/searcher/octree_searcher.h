#pragma once

#include <atlas/searcher/searcher.h>

/**
 * @file octree_searcher.h
 * @brief Named "octree" but implemented as `Searcher`'s ordinary
 *        uniform-grid cell-sort pipeline plus a single-level octant
 *        filter on top of the usual `3x3x3`-cell neighbor search — it
 *        does not build an actual recursive octree.
 *
 * @details
 * `build_neighbors` restricts candidates to those in the *same octant*
 * of the domain relative to its center (each of `x`, `y`, `z` compared
 * against the midpoint independently, giving the 8 octants a real
 * octree's root split would produce) — a single level of octant
 * partitioning applied as a flat per-candidate predicate, not an actual
 * recursive tree of further-subdivided octants. Same legacy-preserving
 * situation as `KdTreeSearcher` (see that file's top-of-file
 * documentation and `docs/updates/updates.md` §2.14): functionally a
 * filtered `SpatialHashingSearcher`.
 */

namespace atlas {

/**
 * @brief Uniform-grid searcher with an added single-level octant filter
 *        on neighbor candidates; see this file's top-of-file
 *        documentation for why this is not an actual octree despite the
 *        name.
 */
class OctreeSearcher final : public Searcher {
public:
    class Builder;

    OctreeSearcher() = default;

    ATLAS_HOST explicit OctreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~OctreeSearcher() override = default;

    /** @brief Runs `Searcher`'s cell-sort pipeline then
     *  `build_neighbors`; skipped if not currently invalidated. */
    ATLAS_HOST void
    build() override;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

public:
    /** @brief Builds the neighbor list restricted to candidates sharing
     *  the same octant as the query particle (see this file's
     *  top-of-file documentation). */
    ATLAS_HOST void
    build_neighbors(int alive, const Vector3* pos);
};

/** @brief Fluent builder for `OctreeSearcher`; requires non-null
 *  universe/fluid. */
class OctreeSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST ATLAS_NODISCARD OctreeSearcher
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<OctreeSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using OctreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher>;

}
