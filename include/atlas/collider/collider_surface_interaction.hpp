#ifndef ATLAS_ENGINE_DEV_COLLIDER_SURFACE_INTERACTION_HPP
#define ATLAS_ENGINE_DEV_COLLIDER_SURFACE_INTERACTION_HPP

#include <atlas/random/random.h>
#include <atlas/random/sampling.h>

namespace atlas::system {

template <typename T>
void
ColliderSurfaceInteraction<T>::set_diffuse_sampling(DiffuseSampling mode) {
    _diffuse_sampling = mode;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_restitution(T restitution_coeff_) {
    _restituion_coeff = restitution_coeff_;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_tangential_momentum_accommodation(T tmac_) {
    _tmac = tmac_;
}

template <typename T>
DiffuseSampling
ColliderSurfaceInteraction<T>::diffuse_sampling() const {
    return _diffuse_sampling;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::restitution() const {
    return _restituion_coeff;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::tangential_momentum_accommodation() const {
    return _tmac;
}

template <typename T>
Vector3<T>
ColliderSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                          const Vector3<T>& normal) const {

    if (_tmac <= T(0)) {
        return math::reflected(incident, normal) * _restituion_coeff;
    }

    const Vector3<T> spec_dir = math::reflected(incident, normal);

    const T u1 = random::rand01(incident);
    const T u2 = random::rand01(normal + incident);

    Vector3<T> diff_dir;
    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {
        diff_dir = random::sample_cosine_hemisphere(normal, u1, u2);
    } else {
        diff_dir = random::sample_uniform_hemisphere(normal, u1, u2);
    }

    const T r = random::rand01(incident + normal * T(17));

    Vector3<T> out_dir;
    if (r < _tmac) {
        out_dir = diff_dir;
    } else {
        out_dir = spec_dir;
    }

    return out_dir * _restituion_coeff;
}

}

#endif