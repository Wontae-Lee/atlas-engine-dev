#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <iterator>
#include <thrust/remove.h>

namespace atlas {

template <ExecutionPolicy P, typename Iterator, typename Predicate>
ATLAS_FORCE_INLINE Iterator
remove_if(Iterator first, Iterator last, Predicate pred) {
    if (first == last) return last;

    // Only the CUDA device policy keeps thrust; every other policy under the TBB
    // build would make nvcc device-compile thrust's illegal temporary-allocator
    // throw (see exclusive_scan.h). Use the STL — parallel for host/device
    // (TBB-backed), sequential for serial.
    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        return thrust::remove_if(thrust::device, first, last, pred);
    } else {
        auto* raw_first = atlas::raw_pointer_cast(&*first);
        auto* raw_last  = raw_first + static_cast<std::ptrdiff_t>(last - first);
        auto* raw_end   = (P == ExecutionPolicy::serial)
              ? std::remove_if(raw_first, raw_last, pred)
              : std::remove_if(std::execution::par, raw_first, raw_last, pred);
        return first + static_cast<std::ptrdiff_t>(raw_end - raw_first);
    }
}

}
