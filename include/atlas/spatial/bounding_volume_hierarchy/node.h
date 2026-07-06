#pragma once

#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas {

struct BVHNode {

    AABB bounds;

    Float3 solid_angle_moment = Float3(0.0f, 0.0f, 0.0f);

    Float3 solid_angle_normal_area = Float3(0.0f, 0.0f, 0.0f);

    float solid_angle_area = 0.0f;

    int left = -1;

    int right = -1;

    int start = -1;

    int count = 0;

    bool is_leaf = false;
};

}
