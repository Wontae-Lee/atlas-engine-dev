#include <atlas/solver/dsmc/dsmc_flatten_workload.h>

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace atlas {

const atlas::DeviceBuffer<int>&
DsmcFlattenWorkload::offsets() const noexcept {
    return collision_offsets;
}

void
DsmcFlattenWorkload::clear() {
    collision_offsets.resize(0);
    collision_cells.resize(0);
    filtered_collision_counts.resize(0);
    total_count_buffer.resize(0);
    flattened_collision_count = 0;
}

bool
DsmcFlattenWorkload::build(int* collision_count_ptr,
                           const int cell_count,
                           const int* allocated_solver_ptr,
                           const int index) {
    if (cell_count <= 0 || collision_count_ptr == nullptr) {
        clear();
        return false;
    }

    if (collision_offsets.size() != static_cast<std::size_t>(cell_count)) {
        collision_offsets.resize(static_cast<std::size_t>(cell_count));
    }

    // When restricted to a solver-allocation subset, zero out every other
    // cell's count into a scratch copy rather than mutating
    // collision_count_ptr in place (which is owned by the caller's
    // UniverseCollisionCountState and must retain its real per-cell
    // counts for whichever solver ends up owning each cell next).
    int* filtered_collision_count_ptr = nullptr;
    if (allocated_solver_ptr != nullptr) {
        if (filtered_collision_counts.size() != static_cast<std::size_t>(cell_count)) {
            filtered_collision_counts.resize(static_cast<std::size_t>(cell_count));
        }

        filtered_collision_count_ptr = atlas::raw_pointer_cast(filtered_collision_counts.data());
        atlas::parallel_for<atlas::ExecutionPolicy::device>(
            0,
            cell_count,
            [=] ATLAS_ALL_DEVICE(const int cell) {
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
        scheduled_collision_count_ptr + cell_count,
        collision_offsets.begin(),
        0);

    const int* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());

    if (total_count_buffer.size() < 1) {
        total_count_buffer.resize(1);
    }
    auto* total_ptr      = atlas::raw_pointer_cast(total_count_buffer.data());
    const auto last_cell = static_cast<std::size_t>(cell_count - 1);
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(int) {
            const int last_offset = collision_offsets_ptr[last_cell];
            const int last_count  = scheduled_collision_count_ptr[last_cell];
            total_ptr[0]          = last_offset + last_count;
        });

    int total_collisions = 0;
    atlas::copy_device_to_host(total_ptr, &total_collisions, 1);

    if (total_collisions <= 0) {
        collision_cells.resize(0);
        flattened_collision_count = 0;
        return false;
    }

    if (total_collisions > std::numeric_limits<int>::max() - 1) {
        throw std::overflow_error("DsmcFlattenWorkload: flattened collision count exceeds int range.");
    }

    if (collision_cells.size() != static_cast<std::size_t>(total_collisions)) {
        collision_cells.resize(static_cast<std::size_t>(total_collisions));
    }
    flattened_collision_count = total_collisions;

    auto* collision_cells_ptr = atlas::raw_pointer_cast(collision_cells.data());

    // Recover each flat candidate's owning cell via binary search over the
    // exclusive-prefix-sum offsets: this is an upper_bound(work_index)
    // search — the largest cell index whose starting offset is still
    // <= work_index. One thread per candidate makes this cheaper than a
    // sequential scan and needs no extra storage beyond the offsets array
    // already built above.
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        total_collisions,
        [=] ATLAS_ALL_DEVICE(const int work_index) {
            int lo = 0;
            int hi = cell_count;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (collision_offsets_ptr[mid] <= work_index) lo = mid + 1;
                else
                    hi = mid;
            }
            collision_cells_ptr[work_index] = lo - 1;
        });

    return true;
}

}
