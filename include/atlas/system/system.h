#pragma once

#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

/**
 * @brief Minimal owning wrapper that couples particle storage with its canonical device probe.
 *
 * @details
 * `System<T>` currently acts as a small composition root for the particle
 * subsystem: it allocates a @ref ParticleData and immediately materializes the
 * single allowed @ref ParticleDeviceProbe used by runtime operators.
 *
 * @tparam T Floating-point scalar type used by particle vectors.
 */
template <typename T>
class System {
public:
    /**
     * @brief Allocate particle storage and create the corresponding device probe.
     *
     * @param buffer_size Number of particle slots to allocate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(size_t buffer_size);
    ~System() = default;

    /**
     * @brief Return shared ownership of the underlying particle data.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDataHostPtr<T>
    particle_data() const noexcept;

    /**
     * @brief Return mutable access to the canonical particle device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>&
    particle_probe() noexcept;

    /**
     * @brief Return const access to the canonical particle device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const ParticleDeviceProbe<T>&
    particle_probe() const noexcept;

private:
    /// @brief Owning particle storage.
    ParticleDataHostPtr<T> _particle_data;
    /// @brief Canonical non-owning probe into @ref _particle_data.
    ParticleDeviceProbe<T> _particle_probe {};
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

} // namespace atlas

#include <atlas/system/system.hpp>
