#pragma once

/**
 * @file sensor_matrics.h
 * @brief Declares observer-side metric storage types used to record source and sink activity.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>

namespace atlas::observer {

/**
 * @brief Abstract base class for all observer metric containers.
 */
class SensorMatrics {
public:
    /**
     * @brief One aggregated metric entry for a single step and unit.
     */
    struct Record {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Record() noexcept = default;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Record(std::size_t step_index_,
                                                             std::size_t unit_index_,
                                                             std::size_t particle_count_) noexcept
            : step_index(step_index_)
            , unit_index(unit_index_)
            , particle_count(particle_count_) { }

        std::size_t step_index     = 0;
        std::size_t unit_index     = 0;
        std::size_t particle_count = 0;
    };

public:
    SensorMatrics()                         = default;
    SensorMatrics(const SensorMatrics&)     = delete;
    SensorMatrics(SensorMatrics&&) noexcept = default;
    virtual ~SensorMatrics()                = default;

    SensorMatrics&
    operator=(const SensorMatrics&)
        = delete;

    SensorMatrics&
    operator=(SensorMatrics&&) noexcept = default;

    /**
     * @brief Returns the number of stored records.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    /**
     * @brief Exports the metric contents to a CSV file inside @p output_directory.
     */
    ATLAS_HOST virtual void
    export_csv(const std::filesystem::path& output_directory) const = 0;
};

/**
 * @brief Common base for step/unit particle-count metrics.
 */
class ParticleCountSensorMatrics : public SensorMatrics {
public:
    ParticleCountSensorMatrics() = default;

    /**
     * @brief Constructs the metrics storage with an initial record capacity hint.
     */
    ATLAS_HOST explicit ParticleCountSensorMatrics(std::size_t reserve_count);

    /**
     * @brief Appends one aggregated record.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    record(std::size_t step_index,
           std::size_t unit_index,
           std::size_t particle_count);

    /**
     * @brief Returns the stored records.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Record>&
    records() const noexcept;

    /**
     * @brief Returns the number of stored records.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

protected:
    /**
     * @brief Returns the CSV filename used by the derived metric type.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual std::string_view
    filename() const noexcept = 0;

    /**
     * @brief Ensures there is enough extra capacity before appending another record.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_extra_capacity(std::size_t additional_records);

protected:
    HostBuffer<Record> _records;
};

/**
 * @brief Aggregated per-step source emission metrics.
 */
class SourceSensorMatrics final : public ParticleCountSensorMatrics {
public:
    using ParticleCountSensorMatrics::ParticleCountSensorMatrics;

    ATLAS_HOST void
    export_csv(const std::filesystem::path& output_directory) const override;

protected:
    ATLAS_HOST ATLAS_NODISCARD std::string_view
    filename() const noexcept override;
};

/**
 * @brief Aggregated per-step sink removal metrics.
 */
class SinkSensorMatrics final : public ParticleCountSensorMatrics {
public:
    using ParticleCountSensorMatrics::ParticleCountSensorMatrics;

    ATLAS_HOST void
    export_csv(const std::filesystem::path& output_directory) const override;

protected:
    ATLAS_HOST ATLAS_NODISCARD std::string_view
    filename() const noexcept override;
};

} // namespace atlas::observer

namespace atlas {

using SensorMatrics       = observer::SensorMatrics;
using SourceSensorMatrics = observer::SourceSensorMatrics;
using SinkSensorMatrics   = observer::SinkSensorMatrics;

} // namespace atlas

#include <atlas/observer/sensor_matrics.hpp>
