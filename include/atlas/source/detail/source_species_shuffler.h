#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/shuffle/shuffle_operator.h>

#include <cstddef>
#include <cstdint>

namespace atlas::detail {

/**
 * @brief Produces per-emission shuffled species assignments.
 */
template <typename T>
class SourceSpeciesShuffler final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shuffle(const DeviceBuffer<std::size_t>& species_cache,
            DeviceBuffer<std::size_t>& shuffled_species,
            DeviceBuffer<std::uint64_t>& shuffle_keys,
            std::uint64_t& shuffle_seed,
            std::size_t count) const;
};

} // namespace atlas::detail

#include <atlas/source/detail/source_species_shuffler.hpp>
