#pragma once

namespace atlas::system {

template <typename T>
ThermometerOperator<T>::ThermometerOperator() noexcept
    : type(ThermometerType::Variance) {
    new (&variance) VarianceThermometerOperator<T> {};
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerType type_) noexcept
    : type(type_) {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T> {};
        return;
    default:
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
ThermometerOperator<T>&
ThermometerOperator<T>::operator=(const ThermometerOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
ThermometerOperator<T>::~ThermometerOperator() noexcept {
    destroy_active();
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const VarianceThermometerOperator<T>& op) noexcept
    : type(ThermometerType::Variance) {
    new (&variance) VarianceThermometerOperator<T>(op);
}

template <typename T>
void
ThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                const SpatialHashingProbe<T>& searcher,
                                const FluidDeviceProbe<T>& particle,
                                const MeasureModeType measure_mode) const {
    switch (type) {
    case ThermometerType::Variance:
        variance.measure(domain, searcher, particle, measure_mode);
        return;
    default:
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}
template <typename T>
void
ThermometerOperator<T>::destroy_active() noexcept {
    switch (type) {
    case ThermometerType::Variance:
        variance.~VarianceThermometerOperator<T>();
        return;
    default:
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
void
ThermometerOperator<T>::copy_from(const ThermometerOperator& other) noexcept {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T>(other.variance);
        return;

    default:
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}
}
