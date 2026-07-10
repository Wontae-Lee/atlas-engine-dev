#pragma once

#include <thrust/device_vector.h>

namespace atlas {

/**
 * @brief Owning container for a contiguous run of @p T stored in device memory.
 *
 * A thin project-wide alias for @c thrust::device_vector<T>: the engine's canonical
 * handle to GPU-resident storage. It owns its allocation and frees it on destruction,
 * resizes with device-side reallocation, and hands out @c thrust::device_ptr iterators
 * that only device execution policies (or the copy helpers in @c atlas/memory/copy.h)
 * may dereference. Its copy constructor and copy assignment are host-only, which is why
 * leaf types that own a @c DeviceBuffer are held move-only through @c HostVariant rather
 * than @c DeviceVariant.
 *
 * @tparam T Element type. Must satisfy thrust's requirements for device storage; for a
 *           buffer that is memcpy'd to/from the host it must be trivially copyable.
 *
 * @note Raw element access from a device kernel goes through @c raw_pointer_cast on
 *       @c begin()/@c data(), not through the container itself, which is host-only.
 */
template <typename T>
using DeviceBuffer = thrust::device_vector<T>;

}