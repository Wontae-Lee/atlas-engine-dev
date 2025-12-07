#ifndef INCLUDE_ATLAS_BUFFER_HOST_BUFFER_H
#define INCLUDE_ATLAS_BUFFER_HOST_BUFFER_H
#ifdef ATLAS_TASKING_CUDA
#include <thrust/host_vector.h>
namespace atlas {
template <typename T>
using HostBuffer = thrust::host_vector<T>;
}
#else
#include <vector>
namespace atlas {
template <typename T>
using HostBuffer = std::vector<T>;
}
#endif
#endif