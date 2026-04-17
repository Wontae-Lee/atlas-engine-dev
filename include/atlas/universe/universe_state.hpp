#pragma once

/**
 * @file universe_state.h
 * @brief Declares universe-side field state types used to store cell-based data.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/vector/vector.h>

#include <cstddef>
#include <typeindex>
#include <utility>

namespace atlas::universe {

/**
 * @brief Type identifier used to index universe states by concrete type.
 */
using TypeId = std::type_index;

/**
 * @brief Abstract base class for all universe-side field states.
 *
 * A UniverseState represents one cell-based attribute field stored in device
 * memory, such as temperature, bulk velocity, momentum weight, thermal energy,
 * or material composition.
 *
 * Concrete derived states own a typed device buffer and expose:
 * - the number of stored cell entries through size()
 * - access to the underlying storage through data()
 *
 * This polymorphic base allows Universe to store heterogeneous field states in
 * a single type-erased registry keyed by concrete type.
 */
class UniverseState {
public:
    /**
     * @brief Default constructor.
     */
    UniverseState() = default;

    /**
     * @brief Copy construction is disabled.
     *
     * Universe states are typically uniquely owned through polymorphic pointers
     * and may manage device-resident buffers.
     */
    UniverseState(const UniverseState&) = delete;

    /**
     * @brief Move constructor.
     */
    UniverseState(UniverseState&&) noexcept = default;

    /**
     * @brief Virtual destructor.
     */
    virtual ~UniverseState() = default;

    /**
     * @brief Copy assignment is disabled.
     *
     * @return Reference to this object.
     */
    UniverseState&
    operator=(const UniverseState&)
        = delete;

    /**
     * @brief Move assignment operator.
     *
     * @return Reference to this object.
     */
    UniverseState&
    operator=(UniverseState&&) noexcept = default;

    /**
     * @brief Returns the number of cell entries stored in this state.
     *
     * This value typically matches the number of cells in the associated universe.
     *
     * @return Number of entries in the underlying buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;
};

/**
 * @brief Universe state storing cell-wise temperature values.
 *
 * Each entry corresponds to the temperature associated with one universe cell.
 *
 * @tparam T Floating-point scalar type used for temperature values.
 */
template <typename T>
class UniverseTemperatureState final : public UniverseState {
public:
    /**
     * @brief Default constructor.
     */
    UniverseTemperatureState() = default;

    /**
     * @brief Constructs a temperature state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseTemperatureState(const std::size_t number_of_cells)
        : _temperature(number_of_cells) { }

    /**
     * @brief Constructs a temperature state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param temperature Device buffer containing cell-wise temperature values.
     */
    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept
        : _temperature(std::move(temperature)) { }

    /**
     * @brief Returns the number of stored temperature entries.
     *
     * @return Number of cell temperature values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _temperature.size();
    }

    /**
     * @brief Returns mutable access to the underlying temperature buffer.
     *
     * @return Reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept {
        return _temperature;
    }

    /**
     * @brief Returns read-only access to the underlying temperature buffer.
     *
     * @return Const reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept {
        return _temperature;
    }

private:
    /**
     * @brief Device buffer storing one temperature value per cell.
     */
    DeviceBuffer<T> _temperature;
};

/**
 * @brief Universe state storing cell-wise bulk velocity vectors.
 *
 * Each entry corresponds to the average or bulk velocity associated with one
 * universe cell.
 *
 * @tparam T Floating-point scalar type used by the vector components.
 */
template <typename T>
class UniverseBulkVelocityState final : public UniverseState {
public:
    /**
     * @brief Default constructor.
     */
    UniverseBulkVelocityState() = default;

    /**
     * @brief Constructs a bulk velocity state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(const std::size_t number_of_cells)
        : _bulk_velocity(number_of_cells) { }

    /**
     * @brief Constructs a bulk velocity state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param bulk_velocity Device buffer containing cell-wise bulk velocity vectors.
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity) noexcept
        : _bulk_velocity(std::move(bulk_velocity)) { }

    /**
     * @brief Returns the number of stored bulk velocity entries.
     *
     * @return Number of cell bulk velocity values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _bulk_velocity.size();
    }

    /**
     * @brief Returns mutable access to the underlying bulk velocity buffer.
     *
     * @return Reference to the bulk velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept {
        return _bulk_velocity;
    }

    /**
     * @brief Returns read-only access to the underlying bulk velocity buffer.
     *
     * @return Const reference to the bulk velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept {
        return _bulk_velocity;
    }

private:
    /**
     * @brief Device buffer storing one bulk velocity vector per cell.
     */
    DeviceBuffer<Vector3<T>> _bulk_velocity;
};

/**
 * @brief Universe state storing cell-wise momentum weights.
 *
 * Each entry corresponds to the accumulated momentum-related weight associated
 * with one universe cell.
 *
 * Depending on the measurement model, this may represent particle count,
 * unit weight accumulation, or another normalization factor used alongside
 * bulk quantities.
 *
 * @tparam T Floating-point scalar type used for weight values.
 */
template <typename T>
class UniverseMomentumWeightState final : public UniverseState {
public:
    /**
     * @brief Default constructor.
     */
    UniverseMomentumWeightState() = default;

    /**
     * @brief Constructs a momentum-weight state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseMomentumWeightState(const std::size_t number_of_cells)
        : _momentum_weight(number_of_cells) { }

    /**
     * @brief Constructs a momentum-weight state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param momentum_weight Device buffer containing cell-wise momentum weights.
     */
    ATLAS_HOST explicit UniverseMomentumWeightState(DeviceBuffer<T> momentum_weight) noexcept
        : _momentum_weight(std::move(momentum_weight)) { }

    /**
     * @brief Returns the number of stored momentum-weight entries.
     *
     * @return Number of cell momentum-weight values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _momentum_weight.size();
    }

    /**
     * @brief Returns mutable access to the underlying momentum-weight buffer.
     *
     * @return Reference to the momentum-weight device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept {
        return _momentum_weight;
    }

    /**
     * @brief Returns read-only access to the underlying momentum-weight buffer.
     *
     * @return Const reference to the momentum-weight device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept {
        return _momentum_weight;
    }

private:
    /**
     * @brief Device buffer storing one momentum-weight value per cell.
     */
    DeviceBuffer<T> _momentum_weight;
};

/**
 * @brief Universe state storing cell-wise thermal energy values.
 *
 * Each entry corresponds to the thermal-energy-related quantity associated with
 * one universe cell.
 *
 * @tparam T Floating-point scalar type used for thermal energy values.
 */
template <typename T>
class UniverseThermalEnergyState final : public UniverseState {
public:
    /**
     * @brief Default constructor.
     */
    UniverseThermalEnergyState() = default;

    /**
     * @brief Constructs a thermal energy state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(const std::size_t number_of_cells)
        : _thermal_energy(number_of_cells) { }

    /**
     * @brief Constructs a thermal energy state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param thermal_energy Device buffer containing cell-wise thermal energy values.
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy) noexcept
        : _thermal_energy(std::move(thermal_energy)) { }

    /**
     * @brief Returns the number of stored thermal energy entries.
     *
     * @return Number of cell thermal energy values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _thermal_energy.size();
    }

    /**
     * @brief Returns mutable access to the underlying thermal energy buffer.
     *
     * @return Reference to the thermal energy device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept {
        return _thermal_energy;
    }

    /**
     * @brief Returns read-only access to the underlying thermal energy buffer.
     *
     * @return Const reference to the thermal energy device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept {
        return _thermal_energy;
    }

private:
    /**
     * @brief Device buffer storing one thermal energy value per cell.
     */
    DeviceBuffer<T> _thermal_energy;
};

/**
 * @brief Universe state storing cell-wise material ratio vectors.
 *
 * Each entry stores an N-dimensional material-ratio vector associated with one cell.
 * This can be used to represent per-cell composition, phase ratios, or mixture
 * proportions across multiple materials.
 *
 * @tparam T Floating-point scalar type used by the ratio vector components.
 * @tparam N Dimension of the material-ratio vector.
 */
template <typename T, std::size_t N>
class UniverseMaterialRatioState final : public UniverseState {
public:
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    /**
     * @brief Default constructor.
     */
    UniverseMaterialRatioState() = default;

    /**
     * @brief Constructs a material-ratio state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseMaterialRatioState(const std::size_t number_of_cells)
        : _material_ratio(number_of_cells) { }

    /**
     * @brief Constructs a material-ratio state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param material_ratio Device buffer containing cell-wise material-ratio vectors.
     */
    ATLAS_HOST explicit UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio) noexcept
        : _material_ratio(std::move(material_ratio)) { }

    /**
     * @brief Returns the number of stored material-ratio entries.
     *
     * @return Number of cell material-ratio vectors.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override {
        return _material_ratio.size();
    }

    /**
     * @brief Returns mutable access to the underlying material-ratio buffer.
     *
     * @return Reference to the material-ratio device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector<T, N>>&
    data() noexcept {
        return _material_ratio;
    }

    /**
     * @brief Returns read-only access to the underlying material-ratio buffer.
     *
     * @return Const reference to the material-ratio device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector<T, N>>&
    data() const noexcept {
        return _material_ratio;
    }

private:
    /**
     * @brief Device buffer storing one material-ratio vector per cell.
     */
    DeviceBuffer<Vector<T, N>> _material_ratio;
};

} // namespace atlas::universe