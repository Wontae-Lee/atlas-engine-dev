#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace atlas::system {

template <typename T>
const atlas::DeviceBuffer<int>&
DsmcFlattenWorkload<T>::offsets() const noexcept {
    return collision_offsets;
}

template <typename T>
void
DsmcFlattenWorkload<T>::clear() {
    collision_offsets.resize(0);
    collision_cells.resize(0);
    filtered_collision_counts.resize(0);
    flattened_collision_count = 0;
}

template <typename T>
bool
DsmcFlattenWorkload<T>::build(int* collision_count_ptr,
                              const int num_of_cells,
                              const int* allocated_solver_ptr,
                              const int index) {
    if (num_of_cells <= 0 || collision_count_ptr == nullptr) {
        clear();
        return false;
    }

    if (collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    int* filtered_collision_count_ptr = nullptr;
    if (allocated_solver_ptr != nullptr) {
        if (filtered_collision_counts.size() != static_cast<std::size_t>(num_of_cells)) {
            filtered_collision_counts.resize(static_cast<std::size_t>(num_of_cells));
        }

        filtered_collision_count_ptr = atlas::raw_pointer_cast(filtered_collision_counts.data());
        atlas::parallel_for<atlas::ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                filtered_collision_count_ptr[cell] = allocated_solver_ptr[cell] == index
                    ? collision_count_ptr[cell]
                    : 0;
            });
    }

    int* scheduled_collision_count_ptr = filtered_collision_count_ptr != nullptr
        ? filtered_collision_count_ptr
        : collision_count_ptr;

    atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        scheduled_collision_count_ptr,
        scheduled_collision_count_ptr + num_of_cells,
        collision_offsets.begin(),
        0);

    const auto last_cell = static_cast<std::size_t>(num_of_cells - 1);
    const int* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());
    int last_offset {};
    int last_count {};
    atlas::copy_device_to_host(collision_offsets_ptr + last_cell, &last_offset, 1);
    atlas::copy_device_to_host(scheduled_collision_count_ptr + last_cell, &last_count, 1);
    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcFlattenWorkload: flattened collision count exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;
    if (total_collisions <= 0) {
        collision_cells.resize(0);
        flattened_collision_count = 0;
        return false;
    }

    if (collision_cells.size() != static_cast<std::size_t>(total_collisions)) {
        collision_cells.resize(static_cast<std::size_t>(total_collisions));
    }
    flattened_collision_count = total_collisions;

    auto* collision_cells_ptr = atlas::raw_pointer_cast(collision_cells.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int offset = collision_offsets_ptr[cell];
            const int count = scheduled_collision_count_ptr[cell];
            for (int local = 0; local < count; ++local) {
                collision_cells_ptr[offset + local] = cell;
            }
        });

    return true;
}

} // namespace atlas::system
