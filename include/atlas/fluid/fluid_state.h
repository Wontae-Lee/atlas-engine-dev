#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>

#include <any>
#include <cstddef>
#include <utility>

namespace atlas {

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

    ATLAS_NODISCARD ATLAS_HOST virtual std::size_t
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
                   const std::size_t kept) {

        if (kept == 0) {
            return;
        }

        using value_type = typename Buffer::value_type;

        if (!_compacted.has_value() || _compacted.type() != typeid(DeviceBuffer<value_type>)) {
            _compacted.emplace<DeviceBuffer<value_type>>();
        }

        auto& compacted = std::any_cast<DeviceBuffer<value_type>&>(_compacted);

        compacted.resize(kept);

        auto* dst          = atlas::raw_pointer_cast(compacted.data());
        auto* src          = atlas::raw_pointer_cast(buffer.data());
        const auto* source = atlas::raw_pointer_cast(compact_indices.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            std::size_t { 0 },
            kept,
            [=] ATLAS_ALL_DEVICE(const std::size_t i) {
                dst[i] = src[source[i]];
            });

        atlas::parallel_for<ExecutionPolicy::device>(
            std::size_t { 0 },
            kept,
            [=] ATLAS_ALL_DEVICE(const std::size_t i) {
                src[i] = dst[i];
            });
    }

    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;

        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }

public:
    std::any _compacted;
};

class FluidPositionState final : public FluidState {
public:
    FluidPositionState() = default;

    ATLAS_HOST explicit FluidPositionState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidPositionState(DeviceBuffer<Float3> position) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _position;
};

class FluidVelocityState final : public FluidState {
public:
    FluidVelocityState() = default;

    ATLAS_HOST explicit FluidVelocityState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidVelocityState(DeviceBuffer<Float3> velocity) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _velocity;
};

class FluidSpeciesState final : public FluidState {
public:
    FluidSpeciesState() = default;

    ATLAS_HOST explicit FluidSpeciesState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<std::size_t>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<std::size_t>&
    data() const noexcept;

private:
    DeviceBuffer<std::size_t> _species;
};

class FluidTemperatureState final : public FluidState {
public:
    FluidTemperatureState() = default;

    ATLAS_HOST explicit FluidTemperatureState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidTemperatureState(DeviceBuffer<float> temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature;
};

class FluidTranslationalEnergyState final : public FluidState {
public:
    FluidTranslationalEnergyState() = default;

    ATLAS_HOST explicit FluidTranslationalEnergyState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidTranslationalEnergyState(DeviceBuffer<float> translational_energy) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _translational_energy;
};

class FluidRotationalEnergyState final : public FluidState {
public:
    FluidRotationalEnergyState() = default;

    ATLAS_HOST explicit FluidRotationalEnergyState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidRotationalEnergyState(DeviceBuffer<float> rotational_energy) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _rotational_energy;
};

class FluidVibrationalEnergyState final : public FluidState {
public:
    FluidVibrationalEnergyState() = default;

    ATLAS_HOST explicit FluidVibrationalEnergyState(std::size_t buffer_size);

    ATLAS_HOST explicit FluidVibrationalEnergyState(DeviceBuffer<float> vibrational_energy) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _vibrational_energy;
};

}