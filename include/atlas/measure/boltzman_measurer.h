#pragma once

/**
 * @file boltzman_measurer.h
 * @brief Declares the BoltzmanMeasurer class for cell-wise thermodynamic field measurement from fluid particle velocities.
 */

#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>

namespace atlas::system {

/**
 * @brief Measures bulk velocity, thermal energy, and temperature fields from fluid particles.
 *
 * This measurer derives from Measurer<T> and uses:
 * - a universe storing cell-based field states,
 * - a fluid storing particle states,
 * - a spatial hashing searcher providing cell-to-particle mapping.
 *
 * The measurement process computes, for each universe cell:
 * - bulk velocity,
 * - particle count,
 * - thermal energy,
 * - temperature derived from the particle velocity distribution.
 *
 * Depending on the configured measure mode, the computed cell temperature may
 * also be written back to particle temperature state.
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class BoltzmanMeasurer final : public Measurer<T> {
public:
    /**
     * @brief Builder for configuring and constructing BoltzmanMeasurer instances.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    BoltzmanMeasurer() = default;

    /**
     * @brief Constructs a BoltzmanMeasurer from its required dependencies.
     *
     * @param universe Host shared pointer to the target universe.
     * @param fluid Host shared pointer to the target fluid.
     * @param searcher Host shared pointer to the spatial hashing searcher.
     * @param measure_mode Mode controlling where measured quantities are written.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    BoltzmanMeasurer(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher,
                     MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    /**
     * @brief Destructor.
     */
    ~BoltzmanMeasurer() override = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent measurer construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Performs the measurement process.
     *
     * This function computes cell-wise bulk velocity, particle count, thermal
     * energy, and temperature from the current fluid particle velocities. When
     * enabled by the measure mode, it also writes cell temperatures back to
     * particle temperature state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure() override;

    /**
     * @brief Returns the current measure mode.
     *
     * @return Measure mode used by this measurer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

private:
    /**
     * @brief Controls how measured quantities are applied.
     */
    MeasureModeType _measure_mode { MeasureModeType::All };
};

/**
 * @brief Builder for BoltzmanMeasurer.
 *
 * This builder collects the required dependencies:
 * - universe,
 * - fluid,
 * - spatial hashing searcher,
 * - measure mode.
 *
 * Validation ensures that all required shared pointers are non-null.
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class BoltzmanMeasurer<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the target universe.
     *
     * @param universe Host shared pointer to the target universe.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the target fluid.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher.
     *
     * @param searcher Host shared pointer to the searcher.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the measure mode.
     *
     * @param measure_mode Measure mode controlling output application.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    /**
     * @brief Builds a validated BoltzmanMeasurer object.
     *
     * @return Constructed BoltzmanMeasurer object.
     *
     * @throw std::runtime_error Thrown if required dependencies are missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE BoltzmanMeasurer<T>
    build() const;

    /**
     * @brief Builds a host-side shared BoltzmanMeasurer object.
     *
     * @return Host shared pointer to a constructed BoltzmanMeasurer object.
     *
     * @throw std::runtime_error Thrown if required dependencies are missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<BoltzmanMeasurer<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Target universe collected by the builder.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Target fluid collected by the builder.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher collected by the builder.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Measure mode selected for the constructed measurer.
     */
    MeasureModeType _measure_mode { MeasureModeType::Field };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::BoltzmanMeasurer.
 *
 * @tparam T Floating-point scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurer = atlas::system::BoltzmanMeasurer<T>;

/**
 * @brief Host-side shared pointer alias for BoltzmanMeasurer.
 *
 * @tparam T Floating-point scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurerHostPtr = atlas::host_shared_ptr<atlas::system::BoltzmanMeasurer<T>>;

/**
 * @brief Device-side shared pointer alias for BoltzmanMeasurer.
 *
 * @tparam T Floating-point scalar type used by the measurer.
 */
template <typename T>
using BoltzmanMeasurerDevicePtr = atlas::device_shared_ptr<atlas::system::BoltzmanMeasurer<T>>;

} // namespace atlas

#include <atlas/measure/boltzman_measurer.hpp>
