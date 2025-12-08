#pragma once
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