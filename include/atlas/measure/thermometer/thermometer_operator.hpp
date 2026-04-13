#pragma once

namespace atlas::system {

template <typename T>
ThermometerOperator<T>::ThermometerOperator() noexcept
    : type(ThermometerType::Average) {
    new (&average) AverageThermometerOperator<T> {};
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerType type_) noexcept
    : type(type_) {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T> {};
        return;
    case ThermometerType::Average:
        new (&average) AverageThermometerOperator<T> {};
        return;
    default:
        type = ThermometerType::Average;
        new (&average) AverageThermometerOperator<T> {};
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
ThermometerOperator<T>::ThermometerOperator(const AverageThermometerOperator<T>& op) noexcept
    : type(ThermometerType::Average) {
    new (&average) AverageThermometerOperator<T>(op);
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
    case ThermometerType::Average:
        average.measure(domain, searcher, particle, measure_mode);
        return;
    }

    average.measure(domain, searcher, particle, measure_mode);
}

template <typename T>
void
ThermometerOperator<T>::destroy_active() noexcept {
    switch (type) {
    case ThermometerType::Variance:
        variance.~VarianceThermometerOperator<T>();
        return;
    case ThermometerType::Average:
        average.~AverageThermometerOperator<T>();
        return;
    }

    average.~AverageThermometerOperator<T>();
}

template <typename T>
void
ThermometerOperator<T>::copy_from(const ThermometerOperator& other) noexcept {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T>(other.variance);
        return;
    case ThermometerType::Average:
        new (&average) AverageThermometerOperator<T>(other.average);
        return;
    default:
        type = ThermometerType::Average;
        new (&average) AverageThermometerOperator<T>(other.average);
        return;
    }
}

}
