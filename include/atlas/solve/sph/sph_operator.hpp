#pragma once

#include <cmath>

namespace atlas::system {

template <typename T>
Poly6SphOperator<T>::Poly6SphOperator(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
Poly6SphOperator<T>::weight(const T distance,
                            const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T x = h * h - distance * distance;
    return x * x * x;
}

template <typename T>
T
Poly6SphOperator<T>::gradient_factor(const T distance,
                                     const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T x = h * h - distance * distance;
    return T(-6) * distance * x * x;
}

template <typename T>
T
Poly6SphOperator<T>::laplacian(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T r2 = distance * distance;
    const T h2 = h * h;
    return T(6) * (h2 - r2) * (T(3) * h2 - T(7) * r2);
}

template <typename T>
SpikySphOperator<T>::SpikySphOperator(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
SpikySphOperator<T>::weight(const T distance,
                            const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T x = h - distance;
    return x * x * x;
}

template <typename T>
T
SpikySphOperator<T>::gradient_factor(const T distance,
                                     const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T x = h - distance;
    return T(-3) * x * x / distance;
}

template <typename T>
T
SpikySphOperator<T>::laplacian(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    return T(6) * (h - distance);
}

template <typename T>
WendlandSphOperator<T>::WendlandSphOperator(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
WendlandSphOperator<T>::weight(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return x * x * x * x * (T(1) + T(4) * q);
}

template <typename T>
T
WendlandSphOperator<T>::gradient_factor(const T distance,
                                        const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return -T(20) * x * x * x / h;
}

template <typename T>
T
WendlandSphOperator<T>::laplacian(const T distance,
                                  const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return T(20) * x * x * (T(4) * q - T(1)) / (h * h);
}

template <typename T>
SphOperator<T>::SphOperator() noexcept
    : type(SphModelType::spiky) {
    new (&spiky) SpikySphOperator<T> {};
}

template <typename T>
SphOperator<T>::SphOperator(const SphModelType type,
                            const T support_scale) noexcept
    : type(type) {
    switch (type) {
    case SphModelType::poly6:
        new (&poly6) Poly6SphOperator<T>(support_scale);
        return;
    case SphModelType::spiky:
        new (&spiky) SpikySphOperator<T>(support_scale);
        return;
    case SphModelType::wendland:
        new (&wendland) WendlandSphOperator<T>(support_scale);
        return;
    default:
        this->type = SphModelType::spiky;
        new (&spiky) SpikySphOperator<T>(support_scale);
        return;
    }
}

template <typename T>
SphOperator<T>::SphOperator(const SphOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
SphOperator<T>&
SphOperator<T>::operator=(const SphOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
SphOperator<T>::~SphOperator() noexcept {
    destroy_active();
}

template <typename T>
SphOperator<T>::SphOperator(const Poly6SphOperator<T>& op)
    : type(SphModelType::poly6) {
    new (&poly6) Poly6SphOperator<T>(op);
}

template <typename T>
SphOperator<T>::SphOperator(const SpikySphOperator<T>& op)
    : type(SphModelType::spiky) {
    new (&spiky) SpikySphOperator<T>(op);
}

template <typename T>
SphOperator<T>::SphOperator(const WendlandSphOperator<T>& op)
    : type(SphModelType::wendland) {
    new (&wendland) WendlandSphOperator<T>(op);
}

template <typename T>
T
SphOperator<T>::weight(const T distance,
                       const T smoothing_length) const noexcept {
    switch (type) {
    case SphModelType::poly6:
        return poly6.weight(distance, smoothing_length);
    case SphModelType::spiky:
        return spiky.weight(distance, smoothing_length);
    case SphModelType::wendland:
        return wendland.weight(distance, smoothing_length);
    default:
        return T(0);
    }
}

template <typename T>
T
SphOperator<T>::gradient_factor(const T distance,
                                const T smoothing_length) const noexcept {
    switch (type) {
    case SphModelType::poly6:
        return poly6.gradient_factor(distance, smoothing_length);
    case SphModelType::spiky:
        return spiky.gradient_factor(distance, smoothing_length);
    case SphModelType::wendland:
        return wendland.gradient_factor(distance, smoothing_length);
    default:
        return T(0);
    }
}

template <typename T>
T
SphOperator<T>::laplacian(const T distance,
                          const T smoothing_length) const noexcept {
    switch (type) {
    case SphModelType::poly6:
        return poly6.laplacian(distance, smoothing_length);
    case SphModelType::spiky:
        return spiky.laplacian(distance, smoothing_length);
    case SphModelType::wendland:
        return wendland.laplacian(distance, smoothing_length);
    default:
        return T(0);
    }
}

template <typename T>
void
SphOperator<T>::destroy_active() noexcept {
    switch (type) {
    case SphModelType::poly6:
        poly6.~Poly6SphOperator<T>();
        return;
    case SphModelType::spiky:
        spiky.~SpikySphOperator<T>();
        return;
    case SphModelType::wendland:
        wendland.~WendlandSphOperator<T>();
        return;
    default:
        spiky.~SpikySphOperator<T>();
        return;
    }
}

template <typename T>
void
SphOperator<T>::copy_from(const SphOperator& other) noexcept {
    switch (type) {
    case SphModelType::poly6:
        new (&poly6) Poly6SphOperator<T>(other.poly6);
        return;
    case SphModelType::spiky:
        new (&spiky) SpikySphOperator<T>(other.spiky);
        return;
    case SphModelType::wendland:
        new (&wendland) WendlandSphOperator<T>(other.wendland);
        return;
    default:
        type = SphModelType::spiky;
        new (&spiky) SpikySphOperator<T>(other.spiky);
        return;
    }
}

}
