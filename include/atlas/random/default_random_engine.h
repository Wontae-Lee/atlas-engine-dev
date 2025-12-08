#pragma once
#ifdef ATLAS_TASKING_CUDA
#include <thrust/random.h>
namespace atlas {
template <typename T>
using default_random_engine = thrust::default_random_engine;
}
#else
#endif