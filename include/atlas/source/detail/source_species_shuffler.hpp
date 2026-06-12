#pragma once

namespace atlas::detail {

template <typename T>
void
SourceSpeciesShuffler<T>::shuffle(const DeviceBuffer<std::size_t>& species_cache,
                                  DeviceBuffer<std::size_t>& shuffled_species,
                                  DeviceBuffer<std::uint64_t>& shuffle_keys,
                                  std::uint64_t& shuffle_seed,
                                  const std::size_t count) const {
    if (count == 0) {
        shuffled_species.clear();
        shuffle_keys.clear();
        return;
    }

    const std::uint64_t seed = shuffle_seed++;

    const auto* cache = atlas::raw_pointer_cast(species_cache.data());
    auto* shuffled    = atlas::raw_pointer_cast(shuffled_species.data());
    auto* keys        = atlas::raw_pointer_cast(shuffle_keys.data());

    const ShuffleOperator shuffle {};

    atlas::parallel_for<ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            shuffled[i] = cache[i];
            keys[i]     = shuffle(static_cast<int>(i), seed);
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        shuffle_keys.begin(),
        shuffle_keys.begin() + static_cast<std::ptrdiff_t>(count),
        shuffled_species.begin());
}

}