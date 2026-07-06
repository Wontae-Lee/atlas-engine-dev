#pragma once

#include <atlas/searcher/searcher.h>

/**
 * @file kdtree_searcher.h
 * @brief Named "k-d tree" but implemented as `Searcher`'s ordinary
 *        uniform-grid cell-sort pipeline plus a single-axis (`x`) band
 *        filter on top of the usual `3x3x3`-cell neighbor search — it
 *        does not build an actual recursive k-d tree.
 *
 * @details
 * `build_neighbors` calls the same `build_cell_neighbors` machinery as
 * `SpatialHashingSearcher`, but with a `CandidateFilter` that further
 * restricts candidates to `|pj.x - pi.x| <= cell_size()`: a coarse
 * approximation of a k-d tree's axis-aligned splitting idea (candidates
 * are additionally filtered along one axis), applied as a flat
 * per-candidate predicate rather than an actual balanced binary tree
 * with recursive median splits. This matches the historical/legacy
 * behavior the engine's non-templated float migration preserved as-is
 * (see `docs/updates/updates.md` §2.14) — `KdTreeSearcher` is
 * functionally a filtered `SpatialHashingSearcher`, not a distinct data
 * structure.
 */

namespace atlas {

/**
 * @brief Uniform-grid searcher with an added x-axis band filter on
 *        neighbor candidates; see this file's top-of-file documentation
 *        for why this is not an actual k-d tree despite the name.
 */
class KdTreeSearcher final : public Searcher {
public:
    class Builder;

    KdTreeSearcher() = default;

    ATLAS_HOST explicit KdTreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~KdTreeSearcher() override = default;

    /** @brief Runs `Searcher`'s cell-sort pipeline then
     *  `build_neighbors`; skipped if not currently invalidated. */
    ATLAS_HOST void
    build() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

public:
    /** @brief Builds the neighbor list restricted to candidates within
     *  `cell_size()` along `x` (see this file's top-of-file
     *  documentation). */
    ATLAS_HOST void
    build_neighbors(int alive, const Float3* pos);
};

/** @brief Fluent builder for `KdTreeSearcher`; requires non-null
 *  universe/fluid. */
class KdTreeSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST KdTreeSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<KdTreeSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using KdTreeSearcherHostPtr = atlas::host_shared_ptr<atlas::Searcher>;

}
