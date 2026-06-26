#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/device_vector.h>
#else
#include <vector>
#endif

namespace atlas {

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using DeviceBuffer = thrust::device_vector<T>;
#else
using DeviceBuffer = std::vector<T>;
#endif

}