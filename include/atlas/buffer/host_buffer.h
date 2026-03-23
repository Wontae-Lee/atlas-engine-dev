#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/host_vector.h>
#else
#include <vector>
#endif

namespace atlas {

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using HostBuffer = thrust::host_vector<T>;
#else
using HostBuffer = std::vector<T>;
#endif

}