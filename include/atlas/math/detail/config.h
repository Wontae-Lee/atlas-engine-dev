#ifndef INCLUDE_ATLAS_MATH_DETAIL_CONFIG_H
#define INCLUDE_ATLAS_MATH_DETAIL_CONFIG_H

#include <atlas/core/macros.h>

#if defined(ATLAS_ENABLE_FAST_MATH)
#define ATLAS_FAST_MATH [[gnu::optimize("fast-math")]]
#else
#define ATLAS_FAST_MATH
#endif
#endif