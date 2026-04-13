#pragma once

namespace atlas::system {

template <typename T>
SphPoly6Kernel<T>::SphPoly6Kernel(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
SphPoly6Kernel<T>::weight(const T distance,
                            const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T x = h * h - distance * distance;
    return x * x * x;
}

template <typename T>
T
SphPoly6Kernel<T>::gradient_factor(const T distance,
                                     const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T x = h * h - distance * distance;
    return T(-6) * distance * x * x;
}

template <typename T>
T
SphPoly6Kernel<T>::laplacian(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T r2 = distance * distance;
    const T h2 = h * h;
    return T(6) * (h2 - r2) * (T(3) * h2 - T(7) * r2);
}

template <typename T>
SphSpikyKernel<T>::SphSpikyKernel(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
SphSpikyKernel<T>::weight(const T distance,
                            const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T x = h - distance;
    return x * x * x;
}

template <typename T>
T
SphSpikyKernel<T>::gradient_factor(const T distance,
                                     const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T x = h - distance;
    return T(-3) * x * x / distance;
}

template <typename T>
T
SphSpikyKernel<T>::laplacian(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    return T(6) * (h - distance);
}

template <typename T>
SphWendlandKernel<T>::SphWendlandKernel(const T support_scale) noexcept
    : support_scale(support_scale) { }

template <typename T>
T
SphWendlandKernel<T>::weight(const T distance,
                               const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return x * x * x * x * (T(1) + T(4) * q);
}

template <typename T>
T
SphWendlandKernel<T>::gradient_factor(const T distance,
                                        const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || !(distance > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return -T(20) * x * x * x / h;
}

template <typename T>
T
SphWendlandKernel<T>::laplacian(const T distance,
                                  const T smoothing_length) const noexcept {
    const T h = smoothing_length * support_scale;
    if (!(h > T(0)) || distance >= h) return T(0);
    const T q = distance / h;
    const T x = T(1) - q;
    return T(20) * x * x * (T(4) * q - T(1)) / (h * h);
}

template <typename T>
SphKernel<T>::SphKernel() noexcept
    : type(SphModelType::spiky) {
    new (&spiky) SphSpikyKernel<T> {};
}

template <typename T>
SphKernel<T>::SphKernel(const SphModelType type,
                                        const T support_scale) noexcept
    : type(type) {
    switch (type) {
    case SphModelType::poly6:
        new (&poly6) SphPoly6Kernel<T>(support_scale);
        return;
    case SphModelType::spiky:
        new (&spiky) SphSpikyKernel<T>(support_scale);
        return;
    case SphModelType::wendland:
        new (&wendland) SphWendlandKernel<T>(support_scale);
        return;
    default:
        this->type = SphModelType::spiky;
        new (&spiky) SphSpikyKernel<T>(support_scale);
        return;
    }
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernel& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
SphKernel<T>&
SphKernel<T>::operator=(const SphKernel& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
SphKernel<T>::~SphKernel() noexcept {
    destroy_active();
}

template <typename T>
SphKernel<T>::SphKernel(const SphPoly6Kernel<T>& op)
    : type(SphModelType::poly6) {
    new (&poly6) SphPoly6Kernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const SphSpikyKernel<T>& op)
    : type(SphModelType::spiky) {
    new (&spiky) SphSpikyKernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const SphWendlandKernel<T>& op)
    : type(SphModelType::wendland) {
    new (&wendland) SphWendlandKernel<T>(op);
}

template <typename T>
T
SphKernel<T>::weight(const T distance,
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
SphKernel<T>::gradient_factor(const T distance,
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
SphKernel<T>::laplacian(const T distance,
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
SphKernel<T>::destroy_active() noexcept {
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
SphKernel<T>::copy_from(const SphKernel& other) noexcept {
    switch (type) {
    case SphModelType::poly6:
        new (&poly6) SphPoly6Kernel<T>(other.poly6);
        return;
    case SphModelType::spiky:
        new (&spiky) SphSpikyKernel<T>(other.spiky);
        return;
    case SphModelType::wendland:
        new (&wendland) SphWendlandKernel<T>(other.wendland);
        return;
    default:
        type = SphModelType::spiky;
        new (&spiky) SphSpikyKernel<T>(other.spiky);
        return;
    }
}

}
