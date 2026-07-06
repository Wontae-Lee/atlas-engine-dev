#pragma once

#include <atlas/searcher/searcher.h>

/**
 * @file spatial_hashing_searcher.h
 * @brief The canonical, unfiltered `Searcher`: cell-sort into the
 *        `Universe`'s uniform grid, kernel-radius neighbor list over
 *        the unfiltered `3x3x3` neighbor-cell block. See `searcher.h`'s
 *        top-of-file documentation for both pipelines this simply runs
 *        as-is with `CandidateFilter` always accepting.
 *
 * This is the base pipeline `KdTreeSearcher`/`OctreeSearcher`/
 * `QuadtreeSearcher` also build on, layering only an extra
 * geometric-partition filter on top of the same cell-sort/neighbor
 * search — see those files for what (and how little) actually
 * distinguishes them from this one.
 */

namespace atlas {

/**
 * @brief Uniform-grid spatial hash searcher: the direct, unfiltered use
 *        of `Searcher`'s cell-sort and neighbor-list machinery. See
 *        this file's top-of-file documentation.
 */
class SpatialHashingSearcher : public Searcher {
public:
    class Builder;

    SpatialHashingSearcher() = default;

    ATLAS_HOST explicit SpatialHashingSearcher(UniverseHostPtr universe, FluidHostPtr fluid);

    ~SpatialHashingSearcher() override = default;

    ATLAS_HOST void
    build() override;

    ATLAS_HOST void
    invalidate() noexcept override;

    ATLAS_HOST void
    reset() noexcept override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /** @brief Public-facing wrapper around `Searcher::prepare_grid_buffers`
     *  (exposes the cell-sort pipeline's individual steps for direct
     *  use/testing, rather than only through `build()`). */
    ATLAS_HOST void
    prepare_buffers(int alive);

    ATLAS_HOST void
    init_indices_iota(int n_active);

    /** @brief Public-facing wrapper around `Searcher::compute_grid_keys`. */
    ATLAS_HOST void
    compute_keys(int alive, const Float3* pos);

    ATLAS_HOST void
    sort_by_key(int active);

    ATLAS_HOST void
    build_cell_ranges(int alive);

    /** @brief Builds the kernel-radius neighbor list with an
     *  always-accepting filter (see this file's top-of-file
     *  documentation) — the unfiltered baseline the other three
     *  searchers restrict further. */
    ATLAS_HOST void
    build_neighbors(int alive, const Float3* pos);

    ATLAS_NODISCARD ATLAS_HOST Float3
    lower_corner() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST Int3
    grid_size() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST float
    inverse_cell_size() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST float
    cell_size() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST const int*
    indices() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_start() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_end() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const int ix, const int iy, const int iz, const Int3& gs) noexcept {
        return Searcher::linear_key(ix, iy, iz, gs);
    }
};

/** @brief Fluent builder for `SpatialHashingSearcher`; requires
 *  non-null universe/fluid. */
class SpatialHashingSearcher::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST SpatialHashingSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SpatialHashingSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};
};

using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<SpatialHashingSearcher>;

}
