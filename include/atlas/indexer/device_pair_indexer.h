#pragma once
#include <atlas/core/macros.h>

namespace atlas {

template <typename Int>
struct DevicePairIndexer {
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int
    pair_index(Int s, Int r, Int n_species) const {

        if (r < s) {
            Int t = s;
            s     = r;
            r     = t;
        }

        return s * n_species
            - (s * (s - 1)) / 2
            + (r - s);
    }
};

}