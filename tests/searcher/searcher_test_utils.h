#pragma once

#include <atlas/fluid/fluid.h>
#include <atlas/memory/copy.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace atlas::test::searcher {

using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::SearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;

inline bool
expect_vec_near(const Vector3& a, const Vector3& b) {
    return std::abs(a.x - b.x) <= atlas::tol
        && std::abs(a.y - b.y) <= atlas::tol
        && std::abs(a.z - b.z) <= atlas::tol;
}

class ExposedSearcher final : public atlas::Searcher {
public:
    using Base = atlas::Searcher;

    ExposedSearcher() = default;

    ExposedSearcher(UniverseHostPtr universe, FluidHostPtr fluid)
        : Base(std::move(universe), std::move(fluid)) {
    }

    void
    build() override {
        const Vector3* positions = position_ptr();
        const int alive          = active_count();

        if (!positions || alive <= 0) {
            reset();
            return;
        }

        prepare_grid_buffers(alive);
        init_indices_iota(alive);
        compute_grid_keys(alive, positions);
        sort_by_key(alive);
        build_cell_ranges(alive);
        clear_neighbors();
        _is_invalidated = false;
    }

    std::size_t
    key_size() const noexcept {
        return _keys.size();
    }

    std::size_t
    index_size() const noexcept {
        return _indices.size();
    }

    std::size_t
    cell_range_size() const noexcept {
        return _cell_start.size();
    }

    std::size_t
    neighbor_offset_size() const noexcept {
        return _neighbor_offsets.size();
    }

    std::size_t
    neighbor_index_size() const noexcept {
        return _neighbor_indices.size();
    }

    bool
    invalidated() const noexcept {
        return _is_invalidated;
    }

    void
    seed_neighbors_for_reset() {
        _neighbor_offsets.resize(2);
        _neighbor_indices.resize(3);
        _neighbor_count = 3;
    }
};

inline UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Vector3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
        .with_cell_size(0.5f)
        .make_host_shared();
}

inline FluidHostPtr
make_fluid(const std::size_t buffer_size = 8) {
    return Fluid::builder()
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

inline FluidHostPtr
make_neighbor_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState>()->data();
    positions[0]    = Vector3(0.10f, 0.10f, 0.10f);
    positions[1]    = Vector3(0.20f, 0.10f, 0.10f);
    positions[2]    = Vector3(0.85f, 0.85f, 0.85f);
    positions[3]    = Vector3(0.95f, 0.85f, 0.85f);

    return fluid;
}

inline FluidHostPtr
make_grid_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState>()->data();
    positions[0]    = Vector3(0.10f, 0.10f, 0.10f);
    positions[1]    = Vector3(0.75f, 0.10f, 0.10f);
    positions[2]    = Vector3(1.20f, 1.20f, 1.20f);
    positions[3]    = Vector3(-0.20f, 0.70f, 0.10f);

    return fluid;
}

inline FluidHostPtr
make_axis_pruning_fluid() {
    auto fluid = make_fluid(3);

    fluid->set_particle_count(3);
    auto& positions = fluid->state<FluidPositionState>()->data();
    positions[0]    = Vector3(0.10f, 0.10f, 0.10f);
    positions[1]    = Vector3(0.55f, 0.10f, 0.10f);
    positions[2]    = Vector3(0.90f, 0.10f, 0.10f);

    return fluid;
}

inline FluidHostPtr
make_quadrant_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState>()->data();
    positions[0]    = Vector3(0.10f, 0.10f, 0.10f);
    positions[1]    = Vector3(0.20f, 0.20f, 0.40f);
    positions[2]    = Vector3(0.70f, 0.20f, 0.10f);
    positions[3]    = Vector3(0.80f, 0.20f, 0.10f);

    return fluid;
}

inline FluidHostPtr
make_octant_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState>()->data();
    positions[0]    = Vector3(0.10f, 0.10f, 0.10f);
    positions[1]    = Vector3(0.20f, 0.20f, 0.20f);
    positions[2]    = Vector3(0.70f, 0.20f, 0.20f);
    positions[3]    = Vector3(0.80f, 0.20f, 0.20f);

    return fluid;
}

template <typename T>
std::vector<T>
copy_values(const T* data, const std::size_t count) {
    std::vector<T> values(count);
    atlas::copy_device_to_host(data, values.data(), values.size());
    return values;
}

inline bool
contains_neighbor(const SearcherHostPtr& searcher, const int particle, const int neighbor) {
    const int* offsets = searcher->neighbor_offsets();
    const int* indices = searcher->neighbor_indices();

    if (!offsets || !indices) {
        return false;
    }

    const std::vector<int> host_offsets = copy_values(offsets + particle, 2);
    const std::vector<int> host_indices = copy_values(indices, static_cast<std::size_t>(searcher->neighbor_count()));

    for (int i = host_offsets[0]; i < host_offsets[1]; ++i) {
        if (host_indices[static_cast<std::size_t>(i)] == neighbor) {
            return true;
        }
    }

    return false;
}

inline int
valid_neighbor_slots(const SearcherHostPtr& searcher, const int particle) {
    const int* offsets = searcher->neighbor_offsets();
    const int* indices = searcher->neighbor_indices();

    if (!offsets || !indices) {
        return 0;
    }

    const std::vector<int> host_offsets = copy_values(offsets + particle, 2);
    const std::vector<int> host_indices = copy_values(indices, static_cast<std::size_t>(searcher->neighbor_count()));

    int count = 0;
    for (int i = host_offsets[0]; i < host_offsets[1]; ++i) {
        if (host_indices[static_cast<std::size_t>(i)] >= 0) {
            ++count;
        }
    }

    return count;
}

}
