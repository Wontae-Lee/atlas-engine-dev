#pragma once

#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/memory/copy.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace atlas::test::searcher {

using atlas::FluidHostPtr;
using atlas::SearcherHostPtr;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::Vector3I;
using atlas::fluid::Fluid;
using atlas::fluid::FluidPositionState;
using atlas::universe::Universe;

class ExposedSearcher final : public atlas::system::Searcher<float> {
public:
    using Base = atlas::system::Searcher<float>;

    ExposedSearcher() = default;

    ExposedSearcher(UniverseHostPtr<float> universe, FluidHostPtr<float> fluid)
        : Base(std::move(universe), std::move(fluid)) {
    }

    void
    build() override {
        const Vector3F* positions = position_ptr();
        const int alive = active_count();

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

inline UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

inline FluidHostPtr<float>
make_fluid(const std::size_t buffer_size = 8) {
    return Fluid<float>::builder()
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

inline FluidHostPtr<float>
make_neighbor_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.20f, 0.10f, 0.10f);
    positions[2] = Vector3F(0.85f, 0.85f, 0.85f);
    positions[3] = Vector3F(0.95f, 0.85f, 0.85f);

    return fluid;
}

inline FluidHostPtr<float>
make_grid_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.75f, 0.10f, 0.10f);
    positions[2] = Vector3F(1.20f, 1.20f, 1.20f);
    positions[3] = Vector3F(-0.20f, 0.70f, 0.10f);

    return fluid;
}

inline FluidHostPtr<float>
make_axis_pruning_fluid() {
    auto fluid = make_fluid(3);

    fluid->set_particle_count(3);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.55f, 0.10f, 0.10f);
    positions[2] = Vector3F(0.90f, 0.10f, 0.10f);

    return fluid;
}

inline FluidHostPtr<float>
make_quadrant_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.20f, 0.20f, 0.40f);
    positions[2] = Vector3F(0.70f, 0.20f, 0.10f);
    positions[3] = Vector3F(0.80f, 0.20f, 0.10f);

    return fluid;
}

inline FluidHostPtr<float>
make_octant_fluid() {
    auto fluid = make_fluid(4);

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.20f, 0.20f, 0.20f);
    positions[2] = Vector3F(0.70f, 0.20f, 0.20f);
    positions[3] = Vector3F(0.80f, 0.20f, 0.20f);

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
contains_neighbor(const SearcherHostPtr<float>& searcher, const int particle, const int neighbor) {
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
valid_neighbor_slots(const SearcherHostPtr<float>& searcher, const int particle) {
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

} // namespace atlas::test::searcher
