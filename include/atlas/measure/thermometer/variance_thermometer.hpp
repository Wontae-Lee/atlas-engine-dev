#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename VarianceThermometer<T>::Builder
VarianceThermometer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
VarianceThermometer<T>::VarianceThermometer(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept
    : _thermometer_operator(thermometer_operator) {
}

template <typename T>
VarianceThermometer<T>::VarianceThermometer(
    const atlas::system::VarianceThermometerOperator<T>& thermometer_operator) noexcept
    : _thermometer_operator(thermometer_operator) {
}

template <typename T>
void
VarianceThermometer<T>::measure(DomainDeviceProbe<T> domain,
                                SpatialHashingProbe<T> searcher,
                                FluidDeviceProbe<T> particle) {
    if (domain.type == DomainType::isothermal) {
        atlas::logger::error()
            << "VarianceThermometer: measure() is forbidden when the domain type is isothermal.";
        throw std::runtime_error("VarianceThermometer: measure() is forbidden when the domain type is isothermal.");
    }

    _thermometer_operator.measure(domain, searcher, particle);
}

template <typename T>
bool
VarianceThermometer<T>::is_valid() const noexcept {
    return _thermometer_operator.type == ThermometerType::Variance;
}

template <typename T>
ThermometerType
VarianceThermometer<T>::type() const noexcept {
    return ThermometerType::Variance;
}

template <typename T>
void
VarianceThermometer<T>::set_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
}

template <typename T>
const atlas::system::ThermometerOperator<T>&
VarianceThermometer<T>::thermometer_operator() const noexcept {
    return _thermometer_operator;
}

template <typename T>
typename VarianceThermometer<T>::Builder&
VarianceThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
    return *this;
}

template <typename T>
typename VarianceThermometer<T>::Builder&
VarianceThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::VarianceThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = atlas::system::ThermometerOperator<T>(thermometer_operator);
    return *this;
}

template <typename T>
VarianceThermometer<T>
VarianceThermometer<T>::Builder::build() const {
    return VarianceThermometer<T>(_thermometer_operator);
}

template <typename T>
atlas::host_shared_ptr<VarianceThermometer<T>>
VarianceThermometer<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<VarianceThermometer<T>>(build());
}

}
