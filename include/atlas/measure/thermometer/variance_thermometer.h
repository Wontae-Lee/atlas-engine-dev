#pragma once

/**
 * @file variance_thermometer.h
 * @brief Declares a concrete variance-based thermometer and its fluent builder.
 *
 * @details
 * This header defines @ref atlas::system::VarianceThermometer, a concrete
 * implementation of the abstract @ref atlas::system::Thermometer interface.
 *
 * The variance thermometer delegates its runtime measurement behavior to a stored
 * @ref atlas::system::ThermometerOperator, whose active implementation is expected
 * to represent variance-based thermometer logic.
 *
 * ## Purpose
 * This class exists to provide a polymorphic, host-constructible thermometer
 * object that can be plugged into the Atlas measurement pipeline while still
 * reusing the lightweight operator-dispatch machinery defined by
 * @ref atlas::system::ThermometerOperator.
 *
 * Typical use cases include:
 * - integrating variance-oriented temperature/statistical measurements into a
 *   system-level measurement stage,
 * - exposing a stable @ref Thermometer interface to higher-level orchestration,
 * - configuring thermometer behavior through a builder or direct operator injection.
 *
 * ## Delegation model
 * `VarianceThermometer` itself does not embed measurement formulas directly in
 * its public interface. Instead, it stores a thermometer operator and forwards
 * calls from @ref measure to that operator.
 *
 * This gives the class:
 * - a polymorphic runtime identity,
 * - a uniform interface shared with other thermometer implementations,
 * - the ability to reuse a compact operator representation internally.
 *
 * ## Measurement mode
 * Construction accepts a @ref MeasureModeType that determines how measurement
 * should be performed when @ref measure is called. This mode is typically stored
 * in the base @ref Thermometer class and then used during dispatch.
 *
 * ## Validity
 * The thermometer exposes @ref is_valid so callers can check whether the stored
 * thermometer state is usable before attempting measurement.
 *
 * ## Construction
 * A variance thermometer may be:
 * - default-constructed,
 * - constructed from a generic @ref ThermometerOperator,
 * - constructed from a concrete @ref VarianceThermometerOperator,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by runtime probes and measurement logic.
 */

#include <atlas/measure/thermometer.h>
#include <atlas/measure/thermometer/thermometer_operator.h>

namespace atlas::system {

/**
 * @brief Concrete thermometer implementation that performs variance-based measurement.
 *
 * @details
 * @ref VarianceThermometer is a final concrete subclass of
 * @ref atlas::system::Thermometer. It provides a polymorphic thermometer object
 * whose runtime behavior is delegated to an internally stored
 * @ref atlas::system::ThermometerOperator.
 *
 * ## Runtime role
 * This class is intended to be used wherever higher-level Atlas code expects a
 * @ref Thermometer<T> object, such as:
 * - measurement pipelines,
 * - system-level orchestration,
 * - host-owned thermometer collections,
 * - builder-configured runtime subsystems.
 *
 * ## Delegated behavior
 * The active measurement logic is stored in @ref _thermometer_operator. When
 * @ref measure is invoked, this object forwards the supplied probes to the
 * stored operator using the current measurement mode.
 *
 * ## Type identity
 * This class represents the variance thermometer runtime type and therefore
 * returns the corresponding @ref ThermometerType from @ref type.
 *
 * ## Validity model
 * A variance thermometer is generally considered valid when its stored operator
 * contains a usable variance thermometer implementation. The exact validation
 * rules are implementation-defined in `variance_thermometer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the measurement pipeline.
 */
template <typename T>
class VarianceThermometer final : public Thermometer<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref VarianceThermometer.
     *
     * @details
     * The builder stages:
     * - the measurement mode,
     * - the thermometer operator,
     * and then constructs either:
     * - a thermometer by value, or
     * - a host-owned shared pointer to a thermometer.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a variance thermometer with default-initialized operator state.
     * The resulting object may or may not be immediately valid, depending on the
     * implementation-defined default operator state.
     */
    VarianceThermometer() = default;

    /**
     * @brief Construct a variance thermometer from a generic thermometer operator.
     *
     * @details
     * This constructor stores the supplied operator and configures the base
     * measurement mode.
     *
     * The caller is responsible for ensuring that the supplied operator is
     * compatible with variance-based measurement semantics.
     *
     * @param thermometer_operator Thermometer operator to store internally.
     * @param measure_mode Measurement mode used when @ref measure is called.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit VarianceThermometer(
        const atlas::system::ThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    /**
     * @brief Construct a variance thermometer from a concrete variance thermometer operator.
     *
     * @details
     * This constructor wraps the supplied concrete variance operator into the
     * internal @ref ThermometerOperator storage and configures the base
     * measurement mode.
     *
     * @param thermometer_operator Concrete variance thermometer operator.
     * @param measure_mode Measurement mode used when @ref measure is called.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit VarianceThermometer(
        const atlas::system::VarianceThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    /**
     * @brief Destructor.
     */
    ~VarianceThermometer() override = default;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Perform variance-based measurement using the stored operator.
     *
     * @details
     * This override forwards the supplied runtime probes to the internally stored
     * thermometer operator together with the currently configured measurement mode.
     *
     * The measurement semantics themselves are defined by the active operator
     * implementation.
     *
     * @param domain Domain probe providing grid and field information.
     * @param searcher Spatial hashing probe providing neighborhood lookup data.
     * @param particle Fluid probe providing particle state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(DomainDeviceProbe<T> domain, SpatialHashingProbe<T> searcher, FluidDeviceProbe<T> particle)
        override;

    /**
     * @brief Return whether this thermometer is in a usable state.
     *
     * @details
     * The exact validity criteria are implementation-defined, but typically
     * reflect whether the stored operator contains a usable variance thermometer
     * configuration.
     *
     * @return `true` if the thermometer can safely perform measurement;
     *         otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the runtime thermometer type tag.
     *
     * @return `ThermometerType::Variance` for this concrete class.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE ThermometerType
    type() const noexcept override;

    /**
     * @brief Replace the internally stored thermometer operator.
     *
     * @details
     * This allows callers to update or swap the delegated measurement behavior
     * after construction.
     *
     * @param thermometer_operator New operator to store.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    /**
     * @brief Return const access to the stored thermometer operator.
     *
     * @return Const reference to the internally stored operator.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::system::ThermometerOperator<T>&
    thermometer_operator() const noexcept;

private:
    /**
     * @brief Internal delegated thermometer operator.
     *
     * @details
     * Stores the runtime variance-measurement logic used by @ref measure.
     */
    atlas::system::ThermometerOperator<T> _thermometer_operator {};
};

/**
 * @brief Fluent builder for @ref VarianceThermometer.
 *
 * @details
 * The builder provides a controlled construction path for
 * @ref VarianceThermometer by staging:
 * - the measurement mode,
 * - the thermometer operator.
 *
 * ## Typical usage
 * @code
 * auto thermometer = atlas::VarianceThermometer<float>::builder()
 *     .with_measure_mode(MeasureModeType::All)
 *     .with_thermometer_operator(atlas::system::VarianceThermometerOperator<float>{})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * The exact validation logic is implementation-defined in
 * `variance_thermometer.hpp`. A typical policy may ensure that the stored
 * operator is compatible with variance-based thermometer semantics.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the measurement pipeline.
 */
template <typename T>
class VarianceThermometer<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - measurement mode set to @ref MeasureModeType::All,
     * - a default-initialized thermometer operator.
     */
    Builder() = default;

    /**
     * @brief Set the measurement mode used by the constructed thermometer.
     *
     * @param measure_mode Measurement mode to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    /**
     * @brief Set the delegated thermometer operator from a generic wrapper.
     *
     * @param thermometer_operator Thermometer operator to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    /**
     * @brief Set the delegated thermometer operator from a concrete variance operator.
     *
     * @param thermometer_operator Concrete variance thermometer operator to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::VarianceThermometerOperator<T>& thermometer_operator) noexcept;

    /**
     * @brief Build a configured @ref VarianceThermometer by value.
     *
     * @return Constructed variance thermometer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE VarianceThermometer<T>
    build() const;

    /**
     * @brief Build a configured @ref VarianceThermometer in host shared ownership.
     *
     * @return `atlas::host_shared_ptr<VarianceThermometer<T>>` owning the constructed thermometer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<VarianceThermometer<T>>
    make_host_shared() const;

private:
    /**
     * @brief Staged measurement mode.
     *
     * @details
     * This mode is forwarded to the constructed thermometer and used when
     * dispatching measurement calls.
     */
    MeasureModeType _measure_mode { MeasureModeType::All };

    /**
     * @brief Staged delegated thermometer operator.
     *
     * @details
     * This operator provides the actual variance-based measurement behavior of
     * the constructed thermometer.
     */
    atlas::system::ThermometerOperator<T> _thermometer_operator {};
};
} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::VarianceThermometer.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using VarianceThermometer = atlas::system::VarianceThermometer<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::VarianceThermometer.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using VarianceThermometerHostPtr = atlas::host_shared_ptr<atlas::system::VarianceThermometer<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::VarianceThermometer.
 *
 * @details
 * This alias is provided for API symmetry with other Atlas component types,
 * even though thermometer orchestration is typically host-driven.
 *
 * @tparam T Floating-point scalar type used by the thermometer.
 */
template <typename T>
using VarianceThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::VarianceThermometer<T>>;

} // namespace atlas

#include <atlas/measure/thermometer/variance_thermometer.hpp>