#pragma once

namespace atlas::detail {

template <typename T>
void
SinkFluidCompactor<T>::compact(const FluidHostPtr<T>& fluid,
                               DeviceBuffer<std::size_t>& keep,
                               DeviceBuffer<std::size_t>& offsets,
                               DeviceBuffer<std::size_t>& compact_indices,
                               DeviceBuffer<std::size_t>& total_count_buffer) const {
    auto* active_state = fluid->template state<atlas::FluidActiveState<T>>();
    if (active_state == nullptr) {
        return;
    }

    auto& active     = active_state->data();
    const auto count = fluid->particle_count();
    if (count == 0) {
        return;
    }

    if (keep.size() != count) {
        keep.resize(count);
    }
    if (offsets.size() != count) {
        offsets.resize(count);
    }
    if (compact_indices.size() != count) {
        compact_indices.resize(count);
    }

    const auto* active_ptr = atlas::raw_pointer_cast(active.data());
    auto* keep_ptr         = atlas::raw_pointer_cast(keep.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        keep.begin(),
        keep.begin() + static_cast<std::ptrdiff_t>(count),
        offsets.begin(),
        std::size_t { 0 });

    if (total_count_buffer.size() < 1) {
        total_count_buffer.resize(1);
    }

    auto* total_ptr          = atlas::raw_pointer_cast(total_count_buffer.data());
    const auto* scan_offsets = atlas::raw_pointer_cast(offsets.data());
    const auto last          = static_cast<std::ptrdiff_t>(count - 1);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_DEVICE(int) {
            total_ptr[0] = scan_offsets[last] + keep_ptr[last];
        });

    std::size_t kept = 0;
    atlas::copy_device_to_host(total_ptr, &kept, 1);

    if (kept == count) {
        fluid->set_particle_count(kept);
        return;
    }

    if (kept == 0) {
        atlas::parallel_fill<ExecutionPolicy::device>(
            active.begin(),
            active.begin() + static_cast<std::ptrdiff_t>(count),
            0);
        fluid->set_particle_count(kept);
        return;
    }

    const auto* offsets_ptr = atlas::raw_pointer_cast(offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(compact_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });

    for (auto& entry : fluid->states()) {
        auto& state = entry.second;
        if (state) {
            state->compact(compact_indices, kept);
        }
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        active.begin() + static_cast<std::ptrdiff_t>(kept),
        active.begin() + static_cast<std::ptrdiff_t>(fluid->buffer_size()),
        0);
    fluid->set_particle_count(kept);
}

}