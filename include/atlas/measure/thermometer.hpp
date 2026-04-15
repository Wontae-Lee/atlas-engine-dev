#pragma once

namespace atlas::system {

template <typename T>
Thermometer<T>::Thermometer(MeasureModeType measure_mode) noexcept
    : _measure_mode(measure_mode) {
}

template <typename T>
void
Thermometer<T>::set_measure_mode(MeasureModeType measure_mode) noexcept {
    _measure_mode = measure_mode;
}

template <typename T>
MeasureModeType
Thermometer<T>::measure_mode() const noexcept {
    return _measure_mode;
}

}