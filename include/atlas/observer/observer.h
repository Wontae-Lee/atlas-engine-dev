#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <filesystem>
#include <utility>

namespace atlas {

class Observer final {
public:
    class Builder;

public:
    Observer()                    = default;
    Observer(const Observer&)     = delete;
    Observer(Observer&&) noexcept = default;
    ~Observer()                   = default;

    Observer&
    operator=(const Observer&)
        = delete;

    Observer&
    operator=(Observer&&) noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    observe(const Fluid& fluid, const Universe& universe, std::size_t step) const;

    ATLAS_HOST void
    resize_counters(std::size_t source_count, std::size_t sink_count, std::size_t species_count);

    ATLAS_HOST void
    reset_counters();

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    species_count() const noexcept {
        return _species_count;
    }

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    spawned() noexcept {
        return _spawned;
    }

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    spawned() const noexcept {
        return _spawned;
    }

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    despawned() noexcept {
        return _despawned;
    }

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    despawned() const noexcept {
        return _despawned;
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    interval() const noexcept {
        return _interval;
    }

    ATLAS_NODISCARD ATLAS_HOST const std::filesystem::path&
    output_directory() const noexcept {
        return _output_directory;
    }

private:
    friend class Builder;

    std::size_t _interval = 0;

    std::filesystem::path _output_directory;

    std::size_t _species_count = 0;

    DeviceBuffer<int> _spawned;

    DeviceBuffer<int> _despawned;
};

class Observer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_interval(std::size_t interval) noexcept;

    ATLAS_HOST Builder&
    with_output_directory(std::filesystem::path output_directory);

    ATLAS_NODISCARD ATLAS_HOST Observer
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    std::size_t _interval = 0;
    std::filesystem::path _output_directory;
};

using ObserverHostPtr = host_shared_ptr<Observer>;

using ObserverDevicePtr = device_shared_ptr<Observer>;

}