#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename AverageThermometer<T>::Builder
AverageThermometer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
AverageThermometer<T>::AverageThermometer(const atlas::system::ThermometerOperator<T>& thermometer_operator,
                                          const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
}

template <typename T>
AverageThermometer<T>::AverageThermometer(
    const atlas::system::AverageThermometerOperator<T>& thermometer_operator,
    const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
}

template <typename T>
void
AverageThermometer<T>::measure(DomainDeviceProbe<T> domain,
                               SpatialHashingProbe<T> searcher,
                               FluidDeviceProbe<T> particle) {
    if (domain.type == DomainType::isothermal) {
        atlas::logger::error()
            << "AverageThermometer: measure() is forbidden when the domain type is isothermal.";
        throw std::runtime_error("AverageThermometer: measure() is forbidden when the domain type is isothermal.");
    }

    _thermometer_operator.measure(domain, searcher, particle, this->measure_mode());
}

template <typename T>
bool
AverageThermometer<T>::is_valid() const noexcept {
    return _thermometer_operator.type == ThermometerType::Average;
}

template <typename T>
ThermometerType
AverageThermometer<T>::type() const noexcept {
    return ThermometerType::Average;
}

template <typename T>
void
AverageThermometer<T>::set_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
}

template <typename T>
const atlas::system::ThermometerOperator<T>&
AverageThermometer<T>::thermometer_operator() const noexcept {
    return _thermometer_operator;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {
    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
    return *this;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::AverageThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = atlas::system::ThermometerOperator<T>(thermometer_operator);
    return *this;
}

template <typename T>
AverageThermometer<T>
AverageThermometer<T>::Builder::build() const {
    return AverageThermometer<T>(_thermometer_operator, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<AverageThermometer<T>>
AverageThermometer<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<AverageThermometer<T>>(build());
}

}
