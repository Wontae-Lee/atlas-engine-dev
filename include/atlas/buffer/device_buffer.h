#ifndef INCLUDE_ATLAS_BUFFER_DEVICE_BUFFER_H
#define INCLUDE_ATLAS_BUFFER_DEVICE_BUFFER_H
#ifdef ATLAS_TASKING_CUDA
#include <thrust/device_vector.h>
namespace atlas {
template <typename T>
using DeviceBuffer = thrust::device_vector<T>;
}
#else
#include <vector>
namespace atlas {
template <typename T>
using DeviceBuffer = std::vector<T>;
}
#endif
#endif