#pragma once

#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas {

template <typename T>
struct BVHNode {

    AABB<T> bounds;

    atlas::Vector<T, 3> solid_angle_moment { T(0), T(0), T(0) };

    atlas::Vector<T, 3> solid_angle_normal_area { T(0), T(0), T(0) };

    T solid_angle_area = T(0);

    int left = -1;

    int right = -1;

    int start = -1;

    int count = 0;

    bool is_leaf = false;
};

}