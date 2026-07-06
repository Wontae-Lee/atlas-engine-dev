#pragma once

#include <thrust/random/uniform_real_distribution.h>

namespace atlas {

template <typename T = float>
using uniform_real_distribution = thrust::uniform_real_distribution<T>;

}
