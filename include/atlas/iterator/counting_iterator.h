#pragma once
#ifdef ATLAS_TASKING_CUDA
#include <thrust/iterator/counting_iterator.h>
namespace atlas {
template <typename T>
using counting_iterator = thrust::counting_iterator<T>;
}
#else
#endif