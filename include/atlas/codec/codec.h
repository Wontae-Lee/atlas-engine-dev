#pragma once

#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Abstract base class for simulation encoding/decoding workflows.
 *
 * This class provides a common interface for components that:
 * - encode simulation data into an intermediate or solver-oriented representation
 * - decode solver results or transformed data back into the fluid/system state
 *
 * A codec instance is bound to:
 * - a universe object describing the simulation domain
 * - a fluid object storing the particle/state data
 * - a spatial hashing searcher used for neighborhood or cell-based indexing support
 *
 * Derived classes are expected to implement the actual encoding and decoding logic
 * by overriding encode() and decode().
 *
 * @tparam T Floating-point scalar type used by the simulation system.
 */
template <typename T>
class Codec {
public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty codec with no attached simulation dependencies.
     * The object is not usable until properly initialized through the
     * non-default constructor or by a derived construction path.
     */
    Codec() = default;

    /**
     * @brief Constructs a codec with its required simulation dependencies.
     *
     * @param domain Host pointer to the universe object.
     * @param fluid Host pointer to the fluid object.
     * @param searcher Host pointer to the spatial hashing searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Codec(UniverseHostPtr<T> domain,
          FluidHostPtr<T> fluid,
          SpatialHashingSearcherHostPtr<T> searcher);

    /**
     * @brief Virtual destructor.
     *
     * Declared virtual because this class is intended to be used polymorphically
     * through base pointers or references.
     */
    virtual ~Codec() = default;

    /**
     * @brief Executes one full codec update cycle.
     *
     * The default update sequence is:
     * 1. encode()
     * 2. decode()
     *
     * Derived classes may override this function if they require a custom update flow,
     * but the default behavior is intended to represent the common encode/decode cycle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update();

    /**
     * @brief Encodes the current simulation state into a derived representation.
     *
     * This function must be implemented by derived classes.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode()
        = 0;

    /**
     * @brief Decodes the derived representation back into simulation state.
     *
     * This function must be implemented by derived classes.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode()
        = 0;

    /**
     * @brief Resets internal codec-owned buffers into a consistent initial state.
     *
     * This function resizes the per-cell solver allocation buffer according to
     * the current universe grid and initializes it to zero.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Returns a mutable reference to the per-cell solver allocation buffer.
     *
     * This buffer is typically used by derived codecs to record which solver,
     * solver block, or computational resource has been assigned to each cell.
     *
     * @return Mutable reference to the allocation buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    allocated_solver() noexcept;

    /**
     * @brief Returns a const reference to the per-cell solver allocation buffer.
     *
     * @return Const reference to the allocation buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    allocated_solver() const noexcept;

protected:
    /**
     * @brief Bound universe object describing the simulation domain and grid.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Bound fluid object containing the simulation state to be processed.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Bound spatial hashing searcher used for cell-based or neighbor-aware processing.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Device buffer storing per-cell solver allocation metadata.
     *
     * The exact semantic meaning depends on the derived codec implementation,
     * but it generally tracks which solver resource is associated with each cell.
     */
    DeviceBuffer<int> d_allocated_solver;
};

}

namespace atlas {

/**
 * @brief Alias for atlas::system::Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Codec = system::Codec<T>;

/**
 * @brief Host shared pointer alias for Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

/**
 * @brief Device shared pointer alias for Codec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

}

#include <atlas/codec/codec.hpp>