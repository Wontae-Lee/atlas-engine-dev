#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/shuffle/shuffle.h>

#include <cstddef>
#include <cstdint>

/**
 * @file source_species_shuffler.h
 * @brief Randomizes which cached candidate spawn slot gets which species
 *        assignment, via the standard parallel "sort by random key"
 *        shuffle.
 *
 * @details
 * `detail::SourceCacheBuilder` assigns species to candidates in a fixed
 * round-robin order (`i % species_count`), which keeps species
 * proportions balanced but means candidate slot `i` always maps to the
 * same species every emission — spatially correlating species with
 * position (since candidates are laid out lattice-order). `shuffle()`
 * breaks that correlation: it copies the cached assignment, tags each
 * entry with an independent hashed key (`Shuffle`, keyed by
 * index and a per-call-incrementing `shuffle_seed`), and sorts the
 * *assignment* array by that random key
 * (`atlas::parallel_sort_by_key`) — this is the standard parallel
 * Fisher-Yates-equivalent shuffle: since the keys are i.i.d. random and
 * independent of position, sorting by them produces a uniformly random
 * permutation of the values, using only a comparison sort rather than a
 * sequential swap-based shuffle (which does not parallelize as
 * directly). The species *distribution* (how many of each species)
 * stays exactly whatever `SourceCacheBuilder` set — only which slot
 * holds which species changes.
 */

namespace atlas::detail {

/** @brief Randomizes species assignment order via sort-by-random-key.
 *  See this file's top-of-file documentation. */
class SourceSpeciesShuffler final {
public:
    /**
     * @brief Copies `species_cache[0, count)` into `shuffled_species`
     *        and randomly permutes it in place, advancing
     *        `shuffle_seed` so each call uses an independent
     *        permutation. No-op (clears both outputs) if `count == 0`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shuffle(const DeviceBuffer<std::size_t>& species_cache,
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

        const Shuffle shuffle {};

        atlas::parallel_for<ExecutionPolicy::device>(
            static_cast<std::size_t>(0),
            count,
            [=] ATLAS_ALL_DEVICE(const std::size_t i) {
                shuffled[i] = cache[i];
                keys[i]     = shuffle(static_cast<int>(i), seed);
            });

        atlas::parallel_sort_by_key<ExecutionPolicy::device>(
            shuffle_keys.begin(),
            shuffle_keys.begin() + static_cast<std::ptrdiff_t>(count),
            shuffled_species.begin());
    }
};

}
