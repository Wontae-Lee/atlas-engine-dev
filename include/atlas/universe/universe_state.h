#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/container/container.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_fill.h>

#include <cstddef>
#include <utility>

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

    ATLAS_NODISCARD ATLAS_HOST virtual std::size_t
    size() const noexcept = 0;

    ATLAS_HOST virtual void
    reset()
        = 0;

protected:
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }
};

class UniverseTemperatureState final : public UniverseState {
public:
    UniverseTemperatureState() = default;

    ATLAS_HOST explicit UniverseTemperatureState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<float> temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature;
};

class UniverseBulkVelocityState final : public UniverseState {
public:
    UniverseBulkVelocityState() = default;

    ATLAS_HOST explicit UniverseBulkVelocityState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Float3> bulk_velocity) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _bulk_velocity;
};

class UniverseFieldForceState final : public UniverseState {
public:
    UniverseFieldForceState() = default;

    ATLAS_HOST explicit UniverseFieldForceState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseFieldForceState(DeviceBuffer<Float3> field_force) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _field_force;
};

class UniverseGravityState final : public UniverseState {
public:
    UniverseGravityState() = default;

    ATLAS_HOST explicit UniverseGravityState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseGravityState(DeviceBuffer<Float3> gravity) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _gravity;
};

class UniverseMaxRelativeSpeedState final : public UniverseState {
public:
    UniverseMaxRelativeSpeedState() = default;

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(DeviceBuffer<float> max_relative_speed) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_relative_speed;
};

class UniverseMaxSigmaGState final : public UniverseState {
public:
    UniverseMaxSigmaGState() = default;

    ATLAS_HOST explicit UniverseMaxSigmaGState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseMaxSigmaGState(DeviceBuffer<float> max_sigma_g) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_sigma_g;
};

class UniverseVolumeState final : public UniverseState {
public:
    UniverseVolumeState() = default;

    ATLAS_HOST explicit UniverseVolumeState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseVolumeState(DeviceBuffer<float> volume) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _volume;
};

class UniverseThermalEnergyState final : public UniverseState {
public:
    UniverseThermalEnergyState() = default;

    ATLAS_HOST explicit UniverseThermalEnergyState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<float> thermal_energy) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _thermal_energy;
};

class UniverseNumberParticleState final : public UniverseState {
public:
    UniverseNumberParticleState() = default;

    ATLAS_HOST explicit UniverseNumberParticleState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseNumberParticleState(DeviceBuffer<float> number_particle) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _number_particle;
};

class UniverseCollisionCountState final : public UniverseState {
public:
    UniverseCollisionCountState() = default;

    ATLAS_HOST explicit UniverseCollisionCountState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseCollisionCountState(DeviceBuffer<int> collision_count) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _collision_count;
};

class UniverseCollisionRemainderState final : public UniverseState {
public:
    UniverseCollisionRemainderState() = default;

    ATLAS_HOST explicit UniverseCollisionRemainderState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseCollisionRemainderState(DeviceBuffer<float> collision_remainder) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _collision_remainder;
};

class UniverseKnudsenNumberState final : public UniverseState {
public:
    UniverseKnudsenNumberState() = default;

    ATLAS_HOST explicit UniverseKnudsenNumberState(std::size_t cell_count);

    ATLAS_HOST explicit UniverseKnudsenNumberState(DeviceBuffer<float> knudsen_number) noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    ATLAS_HOST void
    reset() override;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _knudsen_number;
};

template <std::size_t N>
class UniverseMaterialRatioState final : public UniverseState {
public:
    static_assert(N >= 1, "UniverseMaterialRatioState dimension must be >= 1.");

    using ratio_type = Container<float, N>;

    UniverseMaterialRatioState() = default;

    ATLAS_HOST explicit UniverseMaterialRatioState(const std::size_t cell_count)
        : _material_ratio(cell_count) { }

    ATLAS_HOST explicit UniverseMaterialRatioState(DeviceBuffer<ratio_type> material_ratio) noexcept
        : _material_ratio(std::move(material_ratio)) { }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override {
        return _material_ratio.size();
    }

    ATLAS_HOST void
    reset() override {
        reset_buffer(_material_ratio);
    }

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<ratio_type>&
    data() noexcept {
        return _material_ratio;
    }

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<ratio_type>&
    data() const noexcept {
        return _material_ratio;
    }

private:
    DeviceBuffer<ratio_type> _material_ratio;
};

}
