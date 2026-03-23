#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/random.h>
#else
#include <random>
#endif

namespace atlas {

template <typename T>
#ifdef ATLAS_TASKING_CUDA
using default_random_engine = thrust::default_random_engine;
#else
using default_random_engine = std::default_random_engine;
#endif

}