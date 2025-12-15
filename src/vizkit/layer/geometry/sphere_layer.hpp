#pragma once

#include <algorithm>
#include <cmath>

namespace atlas::vizkit {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
SphereLayer<T>::SphereLayer(const Vector3<T>& center,
                            T radius,
                            int slices,
                            int stacks)
    : GeometryLayer<T>(GL_TRIANGLES)
    , _center(center)
    , _radius(radius)
    , _slices(std::max(3, slices))
    , _stacks(std::max(2, stacks)) {
    static_assert(std::is_same_v<T, float>,
                  "SphereLayer<T> currently supports only T = float.");
}

template <typename T>
void
SphereLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    positions.clear();

    if (_radius <= T(0)) {
        return;
    }


    const T pi     = static_cast<T>(3.14159265358979323846);
    const T two_pi = static_cast<T>(2.0) * pi;

    auto sample = [&](int stack_idx, int slice_idx) -> Vector3<T> {
        const T v   = static_cast<T>(stack_idx) / static_cast<T>(_stacks);
        const T u   = static_cast<T>(slice_idx) / static_cast<T>(_slices);
        const T phi = v * pi;
        const T th  = u * two_pi;

        const T sin_phi = std::sin(phi);
        const T cos_phi = std::cos(phi);
        const T sin_th  = std::sin(th);
        const T cos_th  = std::cos(th);

        const T x = _center.x + _radius * sin_phi * cos_th;
        const T y = _center.y + _radius * cos_phi;
        const T z = _center.z + _radius * sin_phi * sin_th;

        return Vector3<T> { x, y, z };
    };

    positions.reserve(static_cast<std::size_t>(_stacks) * static_cast<std::size_t>(_slices) * 6);


    for (int i = 0; i < _stacks; ++i) {
        const int i_next = i + 1;
        if (i_next > _stacks) continue;

        for (int j = 0; j < _slices; ++j) {
            const int j_next = (j + 1) % _slices;

            Vector3<T> v00 = sample(i, j);
            Vector3<T> v01 = sample(i, j_next);
            Vector3<T> v10 = sample(i_next, j);
            Vector3<T> v11 = sample(i_next, j_next);





            positions.push_back(v00);
            positions.push_back(v10);
            positions.push_back(v11);


            positions.push_back(v00);
            positions.push_back(v11);
            positions.push_back(v01);
        }
    }
}

}