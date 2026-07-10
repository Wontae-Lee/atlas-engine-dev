#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas {

/**
 * @brief Trivially-copyable snapshot of a `SpatialHashingSearcher`'s device arrays.
 *
 * A flat struct of raw device pointers gathered once on the host (via
 * `SpatialHashingSearcher::view()`) so that a `__host__ __device__` lambda can
 * capture the whole searcher by value and read it on the GPU. The umbrella
 * `SpatialHashingSearcher` owns `DeviceBuffer`s whose lifetime is host-only and
 * cannot be captured into a device lambda; this view exposes their contents as
 * bare pointers instead.
 *
 * All four pointers alias buffers owned by the originating searcher and stay
 * valid only until that searcher is re-`classify()`-ed, `reset()`, or destroyed.
 * The view neither owns nor frees anything. A default-constructed view (all
 * null) is the "searcher has not classified yet" state; consumers such as
 * `DsmcSolver::solve` treat any null pointer as "no spatial data available".
 *
 * @note `cell_key` and `sorted_index` are indexed by *sorted particle slot*
 *       (0 .. alive-1); `cell_start` and `cell_end` are indexed by *linear cell
 *       key* (0 .. cell_count-1). A cell's particles occupy the half-open range
 *       `sorted_index[cell_start[key] .. cell_end[key])`.
 */
struct SpatialHashingSearcherView final {

    /// Per-sorted-slot linear cell key, ascending; parallel to `sorted_index`. Device pointer.
    const std::uint32_t* cell_key {};

    /// Original particle indices reordered so equal-cell particles are contiguous. Device pointer.
    const int* sorted_index {};

    /// Per-cell first sorted slot, or -1 if the cell is empty. Indexed by linear cell key. Device pointer.
    const int* cell_start {};

    /// Per-cell one-past-last sorted slot (exclusive), or -1 if empty. Indexed by cell key. Device pointer.
    const int* cell_end {};
};

}