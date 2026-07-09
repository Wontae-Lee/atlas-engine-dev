#pragma once

#include <thrust/device_vector.h>

namespace atlas {

template <typename T>
using DeviceBuffer = thrust::device_vector<T>;

}