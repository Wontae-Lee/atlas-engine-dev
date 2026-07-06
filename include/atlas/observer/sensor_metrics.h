#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>

#include <cstddef>
#include <filesystem>
#include <string_view>

namespace atlas {

class SensorMetrics {
public:
    struct Record {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Record() noexcept = default;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Record(const std::size_t step_index_,
                                                             const std::size_t unit_index_,
                                                             const std::size_t particle_count_) noexcept
            : step_index(step_index_)
            , unit_index(unit_index_)
            , particle_count(particle_count_) { }

        std::size_t step_index     = 0;
        std::size_t unit_index     = 0;
        std::size_t particle_count = 0;
    };

public:
    SensorMetrics()                         = default;
    SensorMetrics(const SensorMetrics&)     = delete;
    SensorMetrics(SensorMetrics&&) noexcept = default;
    virtual ~SensorMetrics()                = default;

    SensorMetrics&
    operator=(const SensorMetrics&)
        = delete;

    SensorMetrics&
    operator=(SensorMetrics&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    ATLAS_HOST virtual void
    export_csv(const std::filesystem::path& output_directory) const = 0;
};

class ParticleCountSensorMetrics : public SensorMetrics {
public:
    ParticleCountSensorMetrics() = default;

    ATLAS_HOST explicit ParticleCountSensorMetrics(std::size_t reserve_count);

    ATLAS_HOST void
    record(std::size_t step_index,
           std::size_t unit_index,
           std::size_t particle_count);

    ATLAS_HOST ATLAS_NODISCARD const HostBuffer<Record>&
    records() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

protected:
    ATLAS_HOST ATLAS_NODISCARD virtual std::string_view
    filename() const noexcept = 0;

    ATLAS_HOST void
    ensure_extra_capacity(std::size_t additional_records);

protected:
    HostBuffer<Record> _records;
};

class SourceSensorMetrics final : public ParticleCountSensorMetrics {
public:
    using ParticleCountSensorMetrics::ParticleCountSensorMetrics;

    ATLAS_HOST void
    export_csv(const std::filesystem::path& output_directory) const override;

protected:
    ATLAS_HOST ATLAS_NODISCARD std::string_view
    filename() const noexcept override;
};

class SinkSensorMetrics final : public ParticleCountSensorMetrics {
public:
    using ParticleCountSensorMetrics::ParticleCountSensorMetrics;

    ATLAS_HOST void
    export_csv(const std::filesystem::path& output_directory) const override;

protected:
    ATLAS_HOST ATLAS_NODISCARD std::string_view
    filename() const noexcept override;
};

}
