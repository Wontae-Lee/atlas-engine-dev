#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {


template <typename T>
typename VarianceThermometer<T>::Builder
VarianceThermometer<T>::builder() noexcept {
    // Return a fresh builder for staged VarianceThermometer construction.
    return Builder {};
}

template <typename T>
VarianceThermometer<T>::VarianceThermometer(
    const atlas::system::ThermometerOperator<T>& thermometer_operator,
    const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
    // Construct a VarianceThermometer from a generic thermometer operator wrapper
    // together with the selected measurement mode.
}

template <typename T>
VarianceThermometer<T>::VarianceThermometer(
    const atlas::system::VarianceThermometerOperator<T>& thermometer_operator,
    const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
    // Construct a VarianceThermometer directly from a specialized
    // VarianceThermometerOperator together with the selected measurement mode.
}

template <typename T>
void
VarianceThermometer<T>::measure(DomainDeviceProbe<T> domain,
                                SpatialHashingProbe<T> searcher,
                                FluidDeviceProbe<T> particle) {
    // Reject measurement on isothermal domains.
    //
    // Rationale:
    // - an isothermal domain has a prescribed fixed temperature field
    // - variance-based measurement would attempt to infer temperature dynamically
    // - that would violate the isothermal-domain contract
    if (domain.type == DomainType::isothermal) {
        atlas::logger::error()
            << "VarianceThermometer: measure() is forbidden when the domain type is isothermal.";
        throw std::runtime_error(
            "VarianceThermometer: measure() is forbidden when the domain type is isothermal.");
    }

    // Delegate the actual work to the stored thermometer operator using the
    // active measurement mode configured in the Thermometer<T> base class.
    _thermometer_operator.measure(domain, searcher, particle, this->measure_mode());
}

template <typename T>
bool
VarianceThermometer<T>::is_valid() const noexcept {
    // This implementation treats the thermometer as always valid.
    //
    // Unlike tag-based validity checks, this returns true unconditionally because
    // the stored operator is assumed to be valid once the object is constructed.
    return true;
}

template <typename T>
ThermometerType
VarianceThermometer<T>::type() const noexcept {
    // Return the runtime type tag for this concrete thermometer.
    return ThermometerType::Variance;
}

template <typename T>
void
VarianceThermometer<T>::set_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    // Replace the currently stored generic thermometer operator.
    _thermometer_operator = thermometer_operator;
}

template <typename T>
const atlas::system::ThermometerOperator<T>&
VarianceThermometer<T>::thermometer_operator() const noexcept {
    // Return the currently stored generic thermometer operator.
    return _thermometer_operator;
}

template <typename T>
typename VarianceThermometer<T>::Builder&
VarianceThermometer<T>::Builder::with_measure_mode(
    const MeasureModeType measure_mode) noexcept {
    // Stage the selected measurement mode inside the builder.
    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
typename VarianceThermometer<T>::Builder&
VarianceThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    // Stage a generic thermometer operator inside the builder.
    _thermometer_operator = thermometer_operator;
    return *this;
}

template <typename T>
typename VarianceThermometer<T>::Builder&
VarianceThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::VarianceThermometerOperator<T>& thermometer_operator) noexcept {
    // Stage a specialized variance thermometer operator inside the builder.
    //
    // It is wrapped into the generic ThermometerOperator<T> holder immediately.
    _thermometer_operator =
        atlas::system::ThermometerOperator<T>(thermometer_operator);
    return *this;
}

template <typename T>
VarianceThermometer<T>
VarianceThermometer<T>::Builder::build() const {
    // Build the final VarianceThermometer from the staged operator and
    // staged measurement mode.
    return VarianceThermometer<T>(_thermometer_operator, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<VarianceThermometer<T>>
VarianceThermometer<T>::Builder::make_host_shared() const {
    // Convenience helper that builds the thermometer and stores it in
    // host-shared memory.
    return atlas::make_host_shared<VarianceThermometer<T>>(build());
}

} // namespace atlas::system