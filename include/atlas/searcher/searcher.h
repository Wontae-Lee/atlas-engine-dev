#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/universe/universe.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace atlas {

template <typename CandidateFilter>
struct SearcherNeighborCount;

template <typename CandidateFilter>
struct SearcherNeighborWrite;

struct SearcherNeighborTotal;

class Searcher {
public:
    Searcher() = default;

    ATLAS_HOST
    Searcher(UniverseHostPtr universe, FluidHostPtr fluid);

    virtual ~Searcher() = default;

    Searcher(const Searcher&) = default;
    Searcher&
    operator=(const Searcher&)
        = default;
    Searcher(Searcher&&) noexcept = default;
    Searcher&
    operator=(Searcher&&) noexcept = default;

    ATLAS_HOST virtual void
    build()
        = 0;

    ATLAS_HOST virtual void
    invalidate() noexcept;

    ATLAS_HOST virtual void
    reset() noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual Float3
    lower_corner() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual Int3
    grid_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    inverse_cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    indices() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    cell_start() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    cell_end() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    neighbor_offsets() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    neighbor_indices() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual int
    neighbor_count() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const int ix, const int iy, const int iz, const Int3& gs) noexcept {
        return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const Int3& cell, const Int3& gs) noexcept {
        return linear_key(cell.x, cell.y, cell.z, gs);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Int3
    cell_for(const Float3& position,
             const Float3& lower_corner,
             const float inverse_cell_size,
             const Int3& grid_size) noexcept {
        const Int3 cell = atlas::to_vector3i(atlas::floor((position - lower_corner) * inverse_cell_size));
        return atlas::clamp(cell, Int3(0, 0, 0), grid_size - Int3(1, 1, 1));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    contains_cell(const Int3& cell, const Int3& grid_size) noexcept {
        return atlas::all(cell >= Int3(0, 0, 0))
            && atlas::all(cell < grid_size);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static int
    search_radius_for(const float length, const float cell_size) noexcept {
        return static_cast<int>(std::ceil(length / cell_size));
    }

protected:
    ATLAS_HOST void
    validate_dependencies(const char* owner) const;

    ATLAS_HOST const Float3*
    position_ptr() const noexcept;

    ATLAS_HOST int
    active_count() const noexcept;

    ATLAS_HOST void
    prepare_grid_buffers(int alive);

public:
    ATLAS_HOST void
    init_indices_iota(int alive);

    ATLAS_HOST void
    compute_grid_keys(int alive, const Float3* positions);

protected:
    ATLAS_HOST void
    sort_by_key(int alive);

public:
    ATLAS_HOST void
    build_cell_ranges(int alive);

protected:
    template <typename CandidateFilter>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_neighbors(const int alive, const Float3* positions, CandidateFilter filter) {
        _neighbor_offsets.resize(static_cast<std::size_t>(alive + 1));
        _neighbor_counts.resize(static_cast<std::size_t>(alive));

        auto* counts        = atlas::raw_pointer_cast(_neighbor_counts.data());
        const auto* indices = atlas::raw_pointer_cast(_indices.data());
        const auto* start   = atlas::raw_pointer_cast(_cell_start.data());
        const auto* end     = atlas::raw_pointer_cast(_cell_end.data());
        const Float3 lc     = _universe->lower_corner();
        const float inv_h   = _universe->inverse_cell_size();
        const float radius2 = _universe->cell_size() * _universe->cell_size();
        const Int3 gs       = _universe->grid_size();

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            alive,
            SearcherNeighborCount<CandidateFilter> {
                counts,
                indices,
                start,
                end,
                positions,
                filter,
                lc,
                inv_h,
                radius2,
                gs });

        atlas::exclusive_scan<ExecutionPolicy::device>(
            _neighbor_counts.begin(),
            _neighbor_counts.begin() + static_cast<std::ptrdiff_t>(alive),
            _neighbor_offsets.begin(),
            0);

        const int total = finalize_neighbor_offsets(alive);
        _neighbor_count = total;
        _neighbor_indices.resize(static_cast<std::size_t>(total));
        if (total == 0) {
            return;
        }

        auto* neighbors     = atlas::raw_pointer_cast(_neighbor_indices.data());
        const auto* offsets = atlas::raw_pointer_cast(_neighbor_offsets.data());
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            alive,
            SearcherNeighborWrite<CandidateFilter> {
                neighbors,
                offsets,
                indices,
                start,
                end,
                positions,
                filter,
                lc,
                inv_h,
                radius2,
                gs });
    }

    ATLAS_HOST int
    finalize_neighbor_offsets(int alive);

protected:
    ATLAS_HOST void
    clear_neighbors();

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    DeviceBuffer<std::uint32_t> _keys;

    DeviceBuffer<int> _indices;

    DeviceBuffer<int> _cell_start;

    DeviceBuffer<int> _cell_end;

    DeviceBuffer<int> _neighbor_offsets;

    DeviceBuffer<int> _neighbor_indices;

    DeviceBuffer<int> _neighbor_counts;

    DeviceBuffer<int> _neighbor_total_count;

    int _neighbor_count {};

    bool _is_invalidated { true };
};

template <typename CandidateFilter>
struct SearcherNeighborCount {
    int* counts {};
    const int* indices {};
    const int* start {};
    const int* end {};
    const Float3* positions {};
    CandidateFilter filter {};
    Float3 lower_corner {};
    float inverse_cell_size {};
    float radius_squared {};
    Int3 grid_size {};

    ATLAS_ALL_DEVICE void
    operator()(const int i) const {
        const Float3 pi = positions[i];
        const Int3 cell = Searcher::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

        int count = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const Int3 neighbor_cell = cell + Int3(dx, dy, dz);
                    if (!Searcher::contains_cell(neighbor_cell, grid_size)) continue;

                    const std::uint32_t key = Searcher::linear_key(neighbor_cell, grid_size);
                    const int first         = start[key];
                    if (first < 0) continue;

                    const int last = end[key];
                    for (int cursor = first; cursor < last; ++cursor) {
                        const int j = indices[cursor];
                        if (j == i) continue;

                        const Float3 pj = positions[j];
                        if (!filter(i, j, pi, pj)) continue;

                        const Float3 delta = pj - pi;
                        if (delta.length_squared() <= radius_squared) {
                            ++count;
                        }
                    }
                }
            }
        }

        counts[i] = count;
    }
};

template <typename CandidateFilter>
struct SearcherNeighborWrite {
    int* neighbors {};
    const int* offsets {};
    const int* indices {};
    const int* start {};
    const int* end {};
    const Float3* positions {};
    CandidateFilter filter {};
    Float3 lower_corner {};
    float inverse_cell_size {};
    float radius_squared {};
    Int3 grid_size {};

    ATLAS_ALL_DEVICE void
    operator()(const int i) const {
        const Float3 pi = positions[i];
        const Int3 cell = Searcher::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

        int write = offsets[i];
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const Int3 neighbor_cell = cell + Int3(dx, dy, dz);
                    if (!Searcher::contains_cell(neighbor_cell, grid_size)) continue;

                    const std::uint32_t key = Searcher::linear_key(neighbor_cell, grid_size);
                    const int first         = start[key];
                    if (first < 0) continue;

                    const int last = end[key];
                    for (int cursor = first; cursor < last; ++cursor) {
                        const int j = indices[cursor];
                        if (j == i) continue;

                        const Float3 pj = positions[j];
                        if (!filter(i, j, pi, pj)) continue;

                        const Float3 delta = pj - pi;
                        if (delta.length_squared() <= radius_squared) {
                            neighbors[write++] = j;
                        }
                    }
                }
            }
        }
    }
};

struct SearcherNeighborTotal {
    int* total {};
    int* offsets {};
    const int* counts {};
    int alive {};
    int last {};

    ATLAS_ALL_DEVICE void
    operator()(int) const {
        total[0]       = offsets[last] + counts[last];
        offsets[alive] = total[0];
    }
};

using SearcherHostPtr = atlas::host_shared_ptr<Searcher>;

using SearcherDevicePtr = atlas::device_shared_ptr<Searcher>;

}
