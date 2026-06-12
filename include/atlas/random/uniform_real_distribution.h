#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random/uniform_real_distribution.h>
#else
#include <random>
#endif

namespace atlas {

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using uniform_real_distribution = thrust::uniform_real_distribution<T>;
#else
using uniform_real_distribution = std::uniform_real_distribution<T>;
#endif

}