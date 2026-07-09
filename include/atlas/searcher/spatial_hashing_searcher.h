#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <cstdint>

namespace atlas {

// Sorts particles into a uniform grid keyed by cell. What it produces is the
// sorted particle order (indices) plus, for every cell, the half-open range
// [cell_start, cell_end) of that order belonging to the cell. Walking a
// neighborhood is left to the consumer, which reads those three arrays.
class SpatialHashingSearcher final {
public:
    class Builder;

public:
    SpatialHashingSearcher() = default;

    ATLAS_HOST
    SpatialHashingSearcher(const Float3& lower_corner,
                           float cell_size,
                           const Int3& grid_size);

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    // Classifies particle_count positions into cells. The cell ranges already
    // hold how many particles each cell received, so it also writes that count
    // into number_particle rather than let a consumer rescan for it — but only
    // the DSMC solver and the Knudsen codec need it, so number_particle is an
    // optional output: a null one, or one sized against a different grid, is
    // skipped. Calling this again rebuilds from scratch; there is no
    // cached-result short circuit.
    ATLAS_HOST void
    classify(const FluidPositionState* positions,
             UniverseNumberParticleState* number_particle,
             int particle_count);

    ATLAS_HOST void
    reset();

    ATLAS_NODISCARD ATLAS_HOST Float3
    lower_corner() const noexcept {
        return _lower_corner;
    }

    ATLAS_NODISCARD ATLAS_HOST Int3
    grid_size() const noexcept {
        return _grid_size;
    }

    ATLAS_NODISCARD ATLAS_HOST float
    cell_size() const noexcept {
        return _cell_size;
    }

    ATLAS_NODISCARD ATLAS_HOST float
    inverse_cell_size() const noexcept {
        return _inverse_cell_size;
    }

    ATLAS_NODISCARD ATLAS_HOST int
    cell_count() const noexcept {
        return _cell_count;
    }

    ATLAS_NODISCARD ATLAS_HOST int
    particle_count() const noexcept {
        return static_cast<int>(_indices.size());
    }

    // Particle indices in cell-sorted order. Cell c owns the slice
    // indices[cell_start[c] .. cell_end[c]); cell_start[c] < 0 means the cell
    // holds no particles.
    ATLAS_NODISCARD ATLAS_HOST const int*
    indices() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_start() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_end() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const int ix, const int iy, const int iz, const Int3& grid_size) noexcept {
        return static_cast<std::uint32_t>(ix + iy * grid_size.x + iz * grid_size.x * grid_size.y);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const Int3& cell, const Int3& grid_size) noexcept {
        return linear_key(cell.x, cell.y, cell.z, grid_size);
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

private:
    ATLAS_HOST void
    compute_keys(int alive, const Float3* positions);

    ATLAS_HOST void
    sort_keys(int alive);

    ATLAS_HOST void
    build_cell_ranges(int alive);

    ATLAS_HOST void
    write_cell_counts(UniverseNumberParticleState* number_particle) const;

private:
    int _cell_count = 1;

    float _cell_size = 1.0f;

    float _inverse_cell_size = 1.0f;

    Float3 _lower_corner { 0.0f, 0.0f, 0.0f };

    Int3 _grid_size { 1, 1, 1 };

    DeviceBuffer<std::uint32_t> _keys;

    DeviceBuffer<int> _indices;

    DeviceBuffer<int> _cell_start;

    DeviceBuffer<int> _cell_end;
};

class SpatialHashingSearcher::Builder final {
public:
    Builder() = default;

    // Convenience: takes lower_corner, cell_size and grid_size off the
    // universe. The universe itself is not retained.
    ATLAS_HOST Builder&
    with_universe(const Universe& universe);

    ATLAS_HOST Builder&
    with_lower_corner(const Float3& lower_corner) noexcept;

    ATLAS_HOST Builder&
    with_cell_size(float cell_size) noexcept;

    ATLAS_HOST Builder&
    with_grid_size(const Int3& grid_size) noexcept;

    ATLAS_NODISCARD ATLAS_HOST SpatialHashingSearcher
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SpatialHashingSearcher>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    float _cell_size = 1.0f;

    Float3 _lower_corner { 0.0f, 0.0f, 0.0f };

    Int3 _grid_size { 1, 1, 1 };
};

using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<SpatialHashingSearcher>;

using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<SpatialHashingSearcher>;

}
