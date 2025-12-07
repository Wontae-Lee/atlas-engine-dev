#ifndef ATLAS_ENGINE_DEV_BOX_LAYER_HPP
#define ATLAS_ENGINE_DEV_BOX_LAYER_HPP

namespace atlas::vizkit {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
BoxLayer<T>::BoxLayer(const Vector3<T>& min_corner, const Vector3<T>& max_corner)
    : GeometryLayer<T>(GL_LINES)
    , _min_corner(min_corner)
    , _max_corner(max_corner) { }

template <typename T>
void
BoxLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {

    const auto& mn = _min_corner;
    const auto& mx = _max_corner;

    Vector3<T> v0 { mn.x, mn.y, mn.z };
    Vector3<T> v1 { mx.x, mn.y, mn.z };
    Vector3<T> v2 { mn.x, mx.y, mn.z };
    Vector3<T> v3 { mx.x, mx.y, mn.z };

    Vector3<T> v4 { mn.x, mn.y, mx.z };
    Vector3<T> v5 { mx.x, mn.y, mx.z };
    Vector3<T> v6 { mn.x, mx.y, mx.z };
    Vector3<T> v7 { mx.x, mx.y, mx.z };

    positions.reserve(24);


    positions.push_back(v0);
    positions.push_back(v1);
    positions.push_back(v1);
    positions.push_back(v3);
    positions.push_back(v3);
    positions.push_back(v2);
    positions.push_back(v2);
    positions.push_back(v0);


    positions.push_back(v4);
    positions.push_back(v5);
    positions.push_back(v5);
    positions.push_back(v7);
    positions.push_back(v7);
    positions.push_back(v6);
    positions.push_back(v6);
    positions.push_back(v4);


    positions.push_back(v0);
    positions.push_back(v4);
    positions.push_back(v1);
    positions.push_back(v5);
    positions.push_back(v2);
    positions.push_back(v6);
    positions.push_back(v3);
    positions.push_back(v7);
}

}

#endif