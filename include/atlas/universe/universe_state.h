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
     * @return Number of entries in the underlying buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;
};

/**
 * @brief Universe state storing cell-wise temperature values.
 *
 * @tparam T Floating-point scalar type used for temperature values.
 */
template <typename T>
class UniverseTemperatureState final : public UniverseState {
public:
    UniverseTemperatureState() = default;

    /**
     * @brief Constructs a temperature state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseTemperatureState(std::size_t number_of_cells);

    /**
     * @brief Constructs a temperature state from an existing device buffer.
     *
     * @param temperature Device buffer containing cell-wise temperature values.
     */
    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept;

    /**
     * @brief Returns the number of stored temperature entries.
     *
     * @return Number of cell temperature values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying temperature buffer.
     *
     * @return Reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying temperature buffer.
     *
     * @return Const reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one temperature value per cell.
     */
    DeviceBuffer<T> _temperature;
};

/**
 * @brief Universe state storing cell-wise bulk velocity vectors.
 *
 * @tparam T Floating-point scalar type used by the vector components.
 */
template <typename T>
class UniverseBulkVelocityState final : public UniverseState {
public:
    UniverseBulkVelocityState() = default;

    /**
     * @brief Constructs a bulk velocity state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(std::size_t number_of_cells);

    /**
     * @brief Constructs a bulk velocity state from an existing device buffer.
     *
     * @param bulk_velocity Device buffer containing cell-wise bulk velocity vectors.
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity) noexcept;

    /**
     * @brief Returns the number of stored bulk velocity entries.
     *
     * @return Number of cell bulk velocity values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying bulk velocity buffer.
     *
     * @return Reference to the bulk velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying bulk velocity buffer.
     *
     * @return Const reference to the bulk velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one bulk velocity vector per cell.
     */
    DeviceBuffer<Vector3<T>> _bulk_velocity;
};

/**
 * @brief Universe state storing cell-wise momentum weights.
 *
 * @tparam T Floating-point scalar type used for weight values.
 */
template <typename T>
class UniverseMomentumWeightState final : public UniverseState {
public:
    UniverseMomentumWeightState() = default;

    /**
     * @brief Constructs a momentum-weight state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseMomentumWeightState(std::size_t number_of_cells);

    /**
     * @brief Constructs a momentum-weight state from an existing device buffer.
     *
     * @param momentum_weight Device buffer containing cell-wise momentum weights.
     */
    ATLAS_HOST explicit UniverseMomentumWeightState(DeviceBuffer<T> momentum_weight) noexcept;

    /**
     * @brief Returns the number of stored momentum-weight entries.
     *
     * @return Number of cell momentum-weight values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying momentum-weight buffer.
     *
     * @return Reference to the momentum-weight device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying momentum-weight buffer.
     *
     * @return Const reference to the momentum-weight device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one momentum-weight value per cell.
     */
    DeviceBuffer<T> _momentum_weight;
};

/**
 * @brief Universe state storing cell-wise thermal energy values.
 *
 * @tparam T Floating-point scalar type used for thermal energy values.
 */
template <typename T>
class UniverseThermalEnergyState final : public UniverseState {
public:
    UniverseThermalEnergyState() = default;

    /**
     * @brief Constructs a thermal energy state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(std::size_t number_of_cells);

    /**
     * @brief Constructs a thermal energy state from an existing device buffer.
     *
     * @param thermal_energy Device buffer containing cell-wise thermal energy values.
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy) noexcept;

    /**
     * @brief Returns the number of stored thermal energy entries.
     *
     * @return Number of cell thermal energy values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying thermal energy buffer.
     *
     * @return Reference to the thermal energy device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying thermal energy buffer.
     *
     * @return Const reference to the thermal energy device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one thermal energy value per cell.
     */
    DeviceBuffer<T> _thermal_energy;
};

/**
/**
 * @brief Universe state storing cell-wise particle-count values.
 *
 * Each entry corresponds to the number of particles assigned to one universe cell.
 *
 * @tparam T Floating-point scalar type used for particle-count values.
 */
template <typename T>
class UniverseNumberParticleState final : public UniverseState {
public:
    UniverseNumberParticleState() = default;

    /**
     * @brief Constructs a particle-count state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseNumberParticleState(std::size_t number_of_cells);

    /**
     * @brief Constructs a particle-count state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param number_particle Device buffer containing cell-wise particle-count values.
     */
    ATLAS_HOST explicit UniverseNumberParticleState(DeviceBuffer<T> number_particle) noexcept;

    /**
     * @brief Returns the number of stored particle-count entries.
     *
     * @return Number of cell particle-count values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying particle-count buffer.
     *
     * @return Reference to the particle-count device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying particle-count buffer.
     *
     * @return Const reference to the particle-count device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one particle-count value per cell.
     */
    DeviceBuffer<T> _number_particle;
};

/**
 * @brief Universe state storing cell-wise Knudsen number values.
 *
 * Each entry corresponds to the Knudsen number associated with one universe cell.
 *
 * @tparam T Floating-point scalar type used for Knudsen number values.
 */
template <typename T>
class UniverseKnudsenNumberState final : public UniverseState {
public:
    UniverseKnudsenNumberState() = default;

    /**
     * @brief Constructs a Knudsen-number state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseKnudsenNumberState(std::size_t number_of_cells);

    /**
     * @brief Constructs a Knudsen-number state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param knudsen_number Device buffer containing cell-wise Knudsen number values.
     */
    ATLAS_HOST explicit UniverseKnudsenNumberState(DeviceBuffer<T> knudsen_number) noexcept;

    /**
     * @brief Returns the number of stored Knudsen-number entries.
     *
     * @return Number of cell Knudsen-number values.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying Knudsen-number buffer.
     *
     * @return Reference to the Knudsen-number device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying Knudsen-number buffer.
     *
     * @return Const reference to the Knudsen-number device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one Knudsen number value per cell.
     */
    DeviceBuffer<T> _knudsen_number;
};

/**
 * @brief Universe state storing cell-wise material ratio vectors.
 *
 * @tparam T Floating-point scalar type used by the ratio vector components.
 * @tparam N Dimension of the material-ratio vector.
 */
template <typename T, std::size_t N>
class UniverseMaterialRatioState final : public UniverseState {
public:
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    UniverseMaterialRatioState() = default;

    /**
     * @brief Constructs a material-ratio state with storage for the given number of cells.
     *
     * @param number_of_cells Number of cell entries to allocate.
     */
    ATLAS_HOST explicit UniverseMaterialRatioState(std::size_t number_of_cells);

    /**
     * @brief Constructs a material-ratio state from an existing device buffer.
     *
     * @param material_ratio Device buffer containing cell-wise material-ratio vectors.
     */
    ATLAS_HOST explicit UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio) noexcept;

    /**
     * @brief Returns the number of stored material-ratio entries.
     *
     * @return Number of cell material-ratio vectors.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Returns mutable access to the underlying material-ratio buffer.
     *
     * @return Reference to the material-ratio device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector<T, N>>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying material-ratio buffer.
     *
     * @return Const reference to the material-ratio device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector<T, N>>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing one material-ratio vector per cell.
     */
    DeviceBuffer<Vector<T, N>> _material_ratio;
};

} // namespace atlas::universe

#include <atlas/universe/universe_state.hpp>
