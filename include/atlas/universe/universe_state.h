#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <cstddef>

namespace atlas {

class UniverseState {
public:
    UniverseState() = default;

    UniverseState(const UniverseState&) = delete;

    UniverseState(UniverseState&&) noexcept = default;

    virtual ~UniverseState() = default;

    UniverseState&
    operator=(const UniverseState&)
        = delete;

    UniverseState&
    operator=(UniverseState&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    ATLAS_HOST virtual void
    reset()
        = 0;

protected:
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer);
};

template <typename T>
class UniverseTemperatureState final : public UniverseState {
public:
    UniverseTemperatureState() = default;

    ATLAS_HOST explicit UniverseTemperatureState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _temperature;
};

template <typename T>
class UniverseBulkVelocityState final : public UniverseState {
public:
    UniverseBulkVelocityState() = default;

    ATLAS_HOST explicit UniverseBulkVelocityState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3<T>> _bulk_velocity;
};

template <typename T>
class UniverseFieldForceState final : public UniverseState {
public:
    UniverseFieldForceState() = default;

    ATLAS_HOST explicit UniverseFieldForceState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseFieldForceState(DeviceBuffer<Vector3<T>> field_force) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3<T>> _field_force;
};

template <typename T>
class UniverseGravityState final : public UniverseState {
public:
    UniverseGravityState() = default;

    ATLAS_HOST explicit UniverseGravityState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseGravityState(DeviceBuffer<Vector3<T>> gravity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3<T>> _gravity;
};

template <typename T>
class UniverseMaxRelativeSpeedState final : public UniverseState {
public:
    UniverseMaxRelativeSpeedState() = default;

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(DeviceBuffer<T> max_relative_speed) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _max_relative_speed;
};

template <typename T>
class UniverseMaxSigmaGState final : public UniverseState {
public:
    UniverseMaxSigmaGState() = default;

    ATLAS_HOST explicit UniverseMaxSigmaGState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseMaxSigmaGState(DeviceBuffer<T> max_sigma_g) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _max_sigma_g;
};

template <typename T>
class UniverseVolumeState final : public UniverseState {
public:
    UniverseVolumeState() = default;

    ATLAS_HOST explicit UniverseVolumeState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseVolumeState(DeviceBuffer<T> volume) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _volume;
};

template <typename T>
class UniverseThermalEnergyState final : public UniverseState {
public:
    UniverseThermalEnergyState() = default;

    ATLAS_HOST explicit UniverseThermalEnergyState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _thermal_energy;
};

template <typename T>
class UniverseNumberParticleState final : public UniverseState {
public:
    UniverseNumberParticleState() = default;

    ATLAS_HOST explicit UniverseNumberParticleState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseNumberParticleState(DeviceBuffer<T> number_particle) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _number_particle;
};

template <typename T>
class UniverseCollisionCountState final : public UniverseState {
public:
    UniverseCollisionCountState() = default;

    ATLAS_HOST explicit UniverseCollisionCountState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseCollisionCountState(DeviceBuffer<T> collision_count) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _collision_count;
};

template <typename T>
class UniverseCollisionRemainderState final : public UniverseState {
public:
    UniverseCollisionRemainderState() = default;

    ATLAS_HOST explicit UniverseCollisionRemainderState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseCollisionRemainderState(DeviceBuffer<T> collision_remainder) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _collision_remainder;
};

template <typename T>
class UniverseKnudsenNumberState final : public UniverseState {
public:
    UniverseKnudsenNumberState() = default;

    ATLAS_HOST explicit UniverseKnudsenNumberState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseKnudsenNumberState(DeviceBuffer<T> knudsen_number) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    DeviceBuffer<T> _knudsen_number;
};

template <typename T, std::size_t N>
class UniverseMaterialRatioState final : public UniverseState {
public:
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    UniverseMaterialRatioState() = default;

    ATLAS_HOST explicit UniverseMaterialRatioState(std::size_t number_of_cells);

    ATLAS_HOST explicit UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector<T, N>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector<T, N>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector<T, N>> _material_ratio;
};

}

#include <atlas/universe/universe_state.hpp>