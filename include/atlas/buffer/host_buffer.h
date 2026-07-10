#pragma once

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/host_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Owning container for a contiguous run of @p T in host memory, laid out to
 *        interoperate cheaply with a @c DeviceBuffer<T>.
 *
 * Under the CUDA backend this is @c thrust::host_vector<T>; under the host backend it is
 * @c std::vector<T>. The interface a caller uses is the same either way, and so is the
 * one property that matters: constructing a @c HostBuffer from a @c DeviceBuffer's
 * iterator pair is a **single bulk copy**, never a per-element transfer.
 *
 * @code
 * const HostBuffer<int> flags(fluid.active().begin(), fluid.active().end());
 * @endcode
 *
 * That is why the alias tracks the backend instead of being @c std::vector everywhere: a
 * @c std::vector built from @c thrust::device_vector iterators would dereference each
 * element on its own, and each dereference is its own @c cudaMemcpy.
 *
 * Because it is a value container its elements must be copyable — which is precisely why
 * move-only leaves (the ones that own a @c DeviceBuffer) cannot live in a @c HostBuffer
 * and are stored as @c HostPtr instead.
 *
 * @tparam T Element type; must be copyable to be stored by value here.
 */
#if defined(ATLAS_BACKEND_CUDA)
template <typename T>
using HostBuffer = thrust::host_vector<T>;
#else
template <typename T>
using HostBuffer = std::vector<T>;
#endif

}
