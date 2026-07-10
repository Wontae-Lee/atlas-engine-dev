#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cstddef>

namespace atlas {

/**
 * @brief Trivially-copyable device view of the fluid columns the DSMC collision kernel
 *        needs.
 *
 * A @c Fluid is a host-only owner of device buffers and cannot be captured by a device
 * lambda. This view gathers, once on the host, the raw device pointers to the velocity
 * and species columns plus the two scalars the collision step needs, into a plain struct
 * of pointers and PODs. Because it is trivially copyable it can be captured by value into
 * a @c __host__ __device__ lambda and dereferenced on the device.
 *
 * The view does not own anything: the pointers alias the fluid's buffers and are only
 * valid while that fluid is alive and its columns are not reallocated (e.g. by
 * @c compact()). Rebuild it after any operation that may move the buffers.
 */
struct FluidDsmcView final {

    Float3* velocity {}; ///< Device pointer to the velocity column (m/s); written by the
                         ///< collision step. Null if the fluid lacks that column.

    const std::size_t* species {}; ///< Device pointer to the read-only species column
                                   ///< (material indices). Null if the column is absent.

    int particle_count {}; ///< Live particle count at view-construction time.

    float statistical_weight {}; ///< Real molecules per simulated particle; positive.

    /**
     * @brief Gathers the device pointers and scalars for @p fluid into a view.
     *
     * Each column pointer is filled only if that column is registered; a missing column
     * leaves the corresponding pointer null (check @c is_complete() before use). Runs on
     * the host.
     *
     * @param fluid The fluid to view; its columns must outlive the returned view.
     * @return The populated view by value.
     */
    ATLAS_NODISCARD ATLAS_HOST static FluidDsmcView
    make(Fluid& fluid) {
        FluidDsmcView view {};

        if (auto* state = fluid.state<FluidVelocityState>(); state != nullptr) {
            view.velocity = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = fluid.state<FluidSpeciesState>(); state != nullptr) {
            view.species = atlas::raw_pointer_cast(state->data().data());
        }

        view.particle_count     = static_cast<int>(fluid.particle_count());
        view.statistical_weight = fluid.statistical_weight();

        return view;
    }

    /**
     * @brief Reports whether both required columns were present.
     * @return @c true if the velocity and species pointers are both non-null, i.e. the
     *         view is safe for the collision kernel to dereference.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    is_complete() const noexcept {
        return velocity != nullptr && species != nullptr;
    }
};

}