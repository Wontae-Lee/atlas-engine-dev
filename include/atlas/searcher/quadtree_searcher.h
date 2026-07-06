#pragma once

#include <atlas/searcher/searcher.h>

/**
 * @file quadtree_searcher.h
 * @brief Named "quadtree" but implemented as `Searcher`'s ordinary
 *        uniform-grid cell-sort pipeline plus a single-level quadrant
 *        (`x`, `y` only) filter on top of the usual `3x3x3`-cell
 *        neighbor search — it does not build an actual recursive
 *        quadtree.
 *
 * @details
 * Identical construction to `OctreeSearcher` but comparing only `x`/`y`
 * against the domain's center (the 2D analog: 4 quadrants instead of 8
 * octants, `z` ignored) — a single level of quadrant partitioning as a
 * flat predicate, not a recursive tree. Same legacy-preserving situation
 * as `KdTreeSearcher`/`OctreeSearcher` (see `docs/updates/updates.md`
 * §2.14): functionally a filtered `SpatialHashingSearcher`.
 */

namespace atlas {

/**
 * @brief Uniform-grid searcher with an added single-level `(x, y)`
 *        quadrant filter on neighbor candidates; see this file's
 *        top-of-file documentation for why this is not an actual
 *        quadtree despite the name.
 */
class QuadtreeSearcher final : public Searcher {
public:
    class Builder;

    QuadtreeSearcher() = default;

    ATLAS_HOST explicit QuadtreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~QuadtreeSearcher() override = default;

    /** @brief Runs `Searcher`'s cell-sort pipeline then
     *  `build_neighbors`; skipped if not currently invalidated. */
    ATLAS_HOST void
    build() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

public:
    /** @brief Builds the neighbor list restricted to candidates sharing
     *  the same `(x, y)` quadrant as the query particle (see this
     *  file's top-of-file documentation). */
    ATLAS_HOST void
    build_neighbors(int alive, const Float3* pos);
};

/** @brief Fluent builder for `QuadtreeSearcher`; requires non-null
 *  universe/fluid. */
class QuadtreeSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST QuadtreeSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<QuadtreeSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using QuadtreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher>;

}
