#pragma once
#include <atlas/core/macros.h>
#if defined(ATLAS_ENABLE_FAST_MATH)
#define ATLAS_FAST_MATH [[gnu::optimize("fast-math")]]
#else
#define ATLAS_FAST_MATH
#endif