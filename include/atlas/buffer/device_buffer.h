#pragma once

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/device_vector.h>
#else
#include <vector>
#endif

namespace atlas {

/**
 * @brief Owning container for a contiguous run of @p T in the parallel backend's memory.
 *
 * Under the CUDA backend this is @c thrust::device_vector<T>: the engine's handle to
 * GPU-resident storage. It owns its allocation, frees it on destruction, resizes with
 * device-side reallocation, and hands out @c thrust::device_ptr iterators that only a
 * device execution policy (or the helpers in @c atlas/memory/copy.h) may dereference.
 *
 * Under the host backend there is no separate device memory, so it is @c std::vector<T>
 * and every "device" operation runs on CPU threads. The distinction survives anyway: a
 * @c DeviceBuffer is still the thing a kernel may not capture, and callers still reach
 * its elements through @c raw_pointer_cast rather than through the container.
 *
 * Its copy constructor and copy assignment are host-only under CUDA, which is why leaf
 * types that own a @c DeviceBuffer are held move-only through @c HostVariant rather than
 * @c DeviceVariant.
 *
 * @tparam T Element type. Under CUDA it must satisfy thrust's requirements for device
 *           storage; for a buffer that is memcpy'd to or from the host it must be
 *           trivially copyable.
 *
 * @note Raw element access from a device kernel goes through @c raw_pointer_cast on
 *       @c begin() or @c data(), not through the container itself, which is host-only.
 */
#if defined(ATLAS_BACKEND_CUDA)
template <typename T>
using DeviceBuffer = thrust::device_vector<T>;
#else
template <typename T>
using DeviceBuffer = std::vector<T>;
#endif

}
