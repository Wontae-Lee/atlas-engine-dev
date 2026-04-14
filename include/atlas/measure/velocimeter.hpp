#pragma once

namespace atlas::system {

template <typename T>
Velocimeter<T>::Velocimeter(MeasureModeType measure_mode) noexcept
    : _measure_mode(measure_mode) {
}

template <typename T>
void
Velocimeter<T>::set_measure_mode(MeasureModeType measure_mode) noexcept {
    _measure_mode = measure_mode;
}

template <typename T>
MeasureModeType
Velocimeter<T>::measure_mode() const noexcept {
    return _measure_mode;
}

} // namespace atlas::system
