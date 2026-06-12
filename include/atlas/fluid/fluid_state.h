#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <any>
#include <cstddef>

namespace atlas {

template <typename T>
struct FluidInternalEnergy final {

    T translational {};

    T rotational {};

    T vibrational {};
};

class FluidState {
public:
    FluidState() = default;

    FluidState(const FluidState&) = delete;

    FluidState(FluidState&&) noexcept = default;

    virtual ~FluidState() = default;

    FluidState&
    operator=(const FluidState&)
        = delete;

    FluidState&
    operator=(FluidState&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    ATLAS_HOST virtual void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept)
        = 0;

    ATLAS_HOST virtual void
    reset()
        = 0;

public:
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compact_buffer(Buffer& buffer,
                   const DeviceBuffer<std::size_t>& compact_indices,
                   std::size_t kept);

    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer);

public:
    std::any _compacted;
};

template <typename T>
class FluidPositionState final : public FluidState {
public:
    FluidPositionState() = default;

    ATLAS_HOST explicit FluidPositionState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidPositionState(DeviceBuffer<Vector3<T>> position) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3<T>> _position;
};

template <typename T>
class FluidVelocityState final : public FluidState {
public:
    FluidVelocityState() = default;

    ATLAS_HOST explicit FluidVelocityState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidVelocityState(DeviceBuffer<Vector3<T>> velocity) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    DeviceBuffer<Vector3<T>> _velocity;
};

template <typename T>
class FluidSpeciesState final : public FluidState {
public:
    FluidSpeciesState() = default;

    ATLAS_HOST explicit FluidSpeciesState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<std::size_t>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<std::size_t>&
    data() const noexcept;

private:
    DeviceBuffer<std::size_t> _species;
};

template <typename T>
class FluidActiveState final : public FluidState {
public:
    FluidActiveState() = default;

    ATLAS_HOST explicit FluidActiveState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidActiveState(DeviceBuffer<int> active) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<int>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _active;
};

template <typename T>
class FluidTemperatureState final : public FluidState {
public:
    FluidTemperatureState() = default;

    ATLAS_HOST explicit FluidTemperatureState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidTemperatureState(DeviceBuffer<T> temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

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
class FluidInternalEnergyState final : public FluidState {
public:
    FluidInternalEnergyState() = default;

    ATLAS_HOST explicit FluidInternalEnergyState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidInternalEnergyState(DeviceBuffer<FluidInternalEnergy<T>> internal_energy) noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<FluidInternalEnergy<T>>&
    data() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<FluidInternalEnergy<T>>&
    data() const noexcept;

private:
    DeviceBuffer<FluidInternalEnergy<T>> _internal_energy;
};

}

#include <atlas/fluid/fluid_state.hpp>