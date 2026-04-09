#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename RmsThermometer<T>::Builder
RmsThermometer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
RmsThermometer<T>::RmsThermometer(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept
    : _thermometer_operator(thermometer_operator) {
}

template <typename T>
RmsThermometer<T>::RmsThermometer(const atlas::system::RmsThermometerOperator<T>& thermometer_operator) noexcept
    : _thermometer_operator(thermometer_operator) {
}

template <typename T>
void
RmsThermometer<T>::measure(DomainDeviceProbe<T> domain,
                           SpatialHashingProbe<T> searcher,
                           FluidDeviceProbe<T> particle) {
    if (domain.type == DomainType::isothermal) {
        atlas::logger::error()
            << "RmsThermometer: measure() is forbidden when the domain type is isothermal.";
        throw std::runtime_error("RmsThermometer: measure() is forbidden when the domain type is isothermal.");
    }

    _thermometer_operator.measure(domain, searcher, particle);
}

template <typename T>
bool
RmsThermometer<T>::is_valid() const noexcept {
    return _thermometer_operator.type == ThermometerType::Rms;
}

template <typename T>
ThermometerType
RmsThermometer<T>::type() const noexcept {
    return ThermometerType::Rms;
}

template <typename T>
void
RmsThermometer<T>::set_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
}

template <typename T>
const atlas::system::ThermometerOperator<T>&
RmsThermometer<T>::thermometer_operator() const noexcept {
    return _thermometer_operator;
}

template <typename T>
typename RmsThermometer<T>::Builder&
RmsThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
    return *this;
}

template <typename T>
typename RmsThermometer<T>::Builder&
RmsThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::RmsThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = atlas::system::ThermometerOperator<T>(thermometer_operator);
    return *this;
}

template <typename T>
RmsThermometer<T>
RmsThermometer<T>::Builder::build() const {
    return RmsThermometer<T>(_thermometer_operator);
}

template <typename T>
atlas::host_shared_ptr<RmsThermometer<T>>
RmsThermometer<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<RmsThermometer<T>>(build());
}

}
