#pragma once

#include <thrust/host_vector.h>

namespace atlas {

/**
 * @brief Owning container for a contiguous run of @p T stored in pinned/pageable host
 *        memory, laid out to interoperate cheaply with a @c DeviceBuffer<T>.
 *
 * A project-wide alias for @c thrust::host_vector<T>. It mirrors @c DeviceBuffer on the
 * CPU side: same interface, host-resident storage, and layout compatible with a
 * device-to-host or host-to-device @c thrust::copy. Because it is a value container its
 * elements must be copyable — that is precisely why move-only leaves (which own a
 * @c DeviceBuffer) cannot live in a @c HostBuffer and are stored as @c HostPtr instead.
 *
 * @tparam T Element type; must be copyable to be stored by value here.
 */
template <typename T>
using HostBuffer = thrust::host_vector<T>;

}