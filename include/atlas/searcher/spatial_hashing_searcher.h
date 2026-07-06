#pragma once

#include <atlas/searcher/searcher.h>

namespace atlas {

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

    ATLAS_HOST void
    prepare_buffers(int alive);

    ATLAS_HOST void
    init_indices_iota(int n_active);

    ATLAS_HOST void
    compute_keys(int alive, const Float3* pos);

    ATLAS_HOST void
    sort_by_key(int active);

    ATLAS_HOST void
    build_cell_ranges(int alive);

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
