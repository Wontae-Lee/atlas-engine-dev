#pragma once
#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::spatial {

template <typename T>
struct BVHNode {

    AABB<T> bounds;

    int left = -1;

    int right = -1;

    int start = -1;

    int count = 0;

    bool is_leaf = false;
};

}
