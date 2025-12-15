#pragma once
#ifdef ATLAS_TASKING_CUDA
#include <thrust/random/uniform_real_distribution.h>

namespace atlas {
template <typename T>
using uniform_real_distribution = thrust::uniform_real_distribution<T>;
}
#else
#endif