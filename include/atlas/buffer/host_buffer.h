#pragma once

#include <thrust/host_vector.h>

namespace atlas {

template <typename T>
using HostBuffer = thrust::host_vector<T>;

}
