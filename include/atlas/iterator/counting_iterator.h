#pragma once

#include <cstddef>
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

template <typename T>
using counting_iterator = thrust::counting_iterator<T, thrust::use_default, thrust::use_default, std::ptrdiff_t>;

}
