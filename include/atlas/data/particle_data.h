#pragma once

#include <cstdint>

#include <atlas/buffer/device_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
namespace atlas {
namespace system {
    /**
     * @brief Non-owning device-side view into particle storage buffers.
     *
     * @details
     * This probe is the lightweight structure passed into kernels and runtime
     * systems. It exposes raw pointers for particle attributes together with the
     * active prefix length (`particle_count`) and total allocated capacity
     * (`buffer_size`).
     *
     * @tparam T Floating-point scalar type used by position and velocity.
     */
    template <typename T>
    struct ParticleDeviceProbe {
        /// @brief Pointer to particle positions.
        Vector3<T>* pos { nullptr };
        /// @brief Pointer to particle velocities.
        Vector3<T>* vel { nullptr };
        /// @brief Pointer to per-particle species identifiers.
        size_t* species { nullptr };
        /// @brief Pointer to active/inactive flags for each allocated slot.
        int* acitve { nullptr };
        /// @brief Number of active particles stored in the prefix of each buffer.
        int particle_count { 0 };
        /// @brief Total allocated number of particle slots.
        size_t buffer_size { 0 };

        /**
         * @brief Return whether the active prefix is empty.
         */
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        empty() const noexcept;

        /**
         * @brief Return whether the probe points at initialized storage.
         */
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        valid() const noexcept;
    };

    /**
     * @brief Owning particle storage for positions, velocities, species, and activity flags.
     *
     * @details
     * `ParticleData<T>` owns the underlying device buffers and can expose exactly
     * one @ref ParticleDeviceProbe through @ref make_device_probe(). Atlas uses
     * the probe to separate owning host-side lifetime from kernel-friendly raw
     * pointer access.
     *
     * @tparam T Floating-point scalar type used by vector-valued attributes.
     */
    template <typename T>
    class ParticleData {
    public:
        /**
         * @brief Allocate particle buffers with a fixed capacity.
         *
         * @param buffer_size Number of slots reserved in every parallel buffer.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE explicit ParticleData(size_t buffer_size);
        ~ParticleData() = default;

        /**
         * @brief Produce the unique device probe exposing raw particle pointers.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>
        make_device_probe() noexcept;

        /**
         * @brief Mutable access to the owned position buffer.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        positions() noexcept;

        /**
         * @brief Mutable access to the owned velocity buffer.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        velocities() noexcept;

        /**
         * @brief Mutable access to the owned species buffer.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<size_t>&
        species() noexcept;

        /**
         * @brief Return the allocated particle capacity.
         */
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
        buffer_size() const noexcept;

        /**
         * @brief Mutable access to the owned activity mask buffer.
         */
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
        active() noexcept;

    private:
        /// @brief Owned position buffer.
        DeviceBuffer<Vector3<T>> d_pos;
        /// @brief Owned velocity buffer.
        DeviceBuffer<Vector3<T>> d_vel;
        /// @brief Owned species-id buffer.
        DeviceBuffer<size_t> d_species;
        /// @brief Owned active/inactive flag buffer.
        DeviceBuffer<int> d_active;
        /// @brief Total allocated slot count shared by every particle buffer.
        size_t _buffer_size        = 0;
        /// @brief Internal guard to enforce a single probe instance.
        std::uint64_t _probe_count = 0;
    };
}

template <typename T>
using ParticleData = system::ParticleData<T>;
template <typename T>
using ParticleDataHostPtr = atlas::host_shared_ptr<system::ParticleData<T>>;

} // namespace atlas

#include <atlas/data/particle_data.hpp>
