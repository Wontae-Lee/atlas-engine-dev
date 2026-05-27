#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cstdint>

namespace atlas::system {

class PiclasPairing final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    select_pair_offsets(int& lhs_local,
                        int& rhs_local,
                        int local_pair,
                        int count,
                        int selector,
                        std::uint64_t seed) noexcept {
        lhs_local = -1;
        rhs_local = -1;

        if (count < 2 || local_pair < 0 || local_pair >= count / 2) {
            return;
        }

        const int offset = atlas::sampling::sample_hashed_index(
            selector,
            count,
            seed + atlas::seed::DSMC_COLLISION_LHS_SALT);

        int stride = count == 2
            ? 1
            : 1 + atlas::sampling::sample_hashed_index(
                  selector,
                  count - 1,
                  seed + atlas::seed::DSMC_COLLISION_RHS_SALT);

        for (;;) {
            int a = stride;
            int b = count;
            while (b != 0) {
                const int next = a % b;
                a = b;
                b = next;
            }

            if (a == 1) {
                break;
            }

            ++stride;
            if (stride >= count) {
                stride = 1;
            }
        }

        const auto lhs_offset = static_cast<std::int64_t>(2 * local_pair) * static_cast<std::int64_t>(stride);
        const auto rhs_offset = static_cast<std::int64_t>(2 * local_pair + 1) * static_cast<std::int64_t>(stride);
        lhs_local = static_cast<int>((static_cast<std::int64_t>(offset) + lhs_offset) % count);
        rhs_local = static_cast<int>((static_cast<std::int64_t>(offset) + rhs_offset) % count);
    }
};

} // namespace atlas::system

namespace atlas {

using PiclasPairing = atlas::system::PiclasPairing;

} // namespace atlas
