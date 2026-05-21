#pragma once

/**
 * @file codec.h
 * @brief Declares the abstract Codec base class for encode/decode solver-allocation workflows.
 *
 * A codec is responsible for translating measured universe/fluid state into
 * per-cell solver-allocation metadata and, if needed, decoding codec-side
 * results back into the simulation state.
 */

#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Abstract base class for codec-driven simulation classification and decoding.
 *
 * `Codec` binds together the simulation dependencies required by derived codec
 * implementations:
 *
 * - a universe, used for cell count, cell volume, and optional cell-level states,
 * - a fluid, used for particle count and statistical weight,
 * - a spatial hashing searcher, used for sorted particle indices and cell ranges.
 *
 * The base class owns `d_allocated_solver`, a device buffer with one integer
 * entry per universe cell. Derived classes typically write this buffer during
 * @ref encode to classify each cell or assign an appropriate solver. The
 * orchestrator later passes this allocation buffer to codec-aware solvers.
 *
 * The default @ref update implementation executes:
 *
 * @code
 * encode();
 * decode();
 * @endcode
 *
 * Derived classes must implement @ref encode and @ref decode.
 *
 * @tparam T Scalar type used by the simulation.
 */
template <typename T>
class Codec {
public:
    /**
     * @brief Raw-pointer view over common codec input/output data.
     *
     * `CodecProbe` is populated by @ref make_probe. It groups optional universe
     * state pointers, codec-owned solver-allocation data, searcher cell ranges,
     * and scalar simulation metadata into a compact object suitable for device
     * kernel capture.
     *
     * The universe-state pointers are optional:
     *
     * - `temperature_ptr` is populated when `UniverseTemperatureState<T>` exists,
     * - `number_particle_ptr` is populated when `UniverseNumberParticleState<T>` exists,
     * - `knudsen_number_ptr` is populated when `UniverseKnudsenNumberState<T>` exists.
     *
     * Searcher pointers are copied from the configured spatial hashing searcher.
     * The current implementation only requires `num_of_cells > 0` for the probe
     * to report success, so derived codecs should check any optional pointer they
     * require before dereferencing it.
     */
    struct CodecProbe {
        /**
         * @brief Raw pointer to optional per-cell temperature data.
         *
         * Points to `UniverseTemperatureState<T>::data()` when that state exists;
         * otherwise remains `nullptr`.
         */
        const T* temperature_ptr {};

        /**
         * @brief Raw pointer to optional per-cell particle-count data.
         *
         * Points to `UniverseNumberParticleState<T>::data()` when that state
         * exists; otherwise remains `nullptr`.
         */
        const T* number_particle_ptr {};

        /**
         * @brief Raw pointer to optional per-cell Knudsen-number output/input data.
         *
         * Points to `UniverseKnudsenNumberState<T>::data()` when that state
         * exists; otherwise remains `nullptr`.
         */
        T* knudsen_number_ptr {};

        /**
         * @brief Raw pointer to the codec-owned per-cell solver allocation buffer.
         *
         * Points to @ref d_allocated_solver when it is non-empty; otherwise
         * remains `nullptr`.
         */
        int* allocated_solver_ptr {};

        /**
         * @brief Raw pointer to the codec-owned fixed solver-index buffer.
         *
         * Fixed-region cells use this solver index during decode.
         */
        const int* fixed_solver_ptr {};

        /**
         * @brief Raw pointer to the codec-owned fixed-region mask.
         *
         * A value of `1` marks a cell whose encode/decode classification should
         * be fixed rather than recomputed by a derived codec.
         */
        const int* fixed_region_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * For each cell, `[cell_start_ptr[cell], cell_end_ptr[cell])` indexes
         * into this array.
         */
        const int* indices_ptr {};

        /**
         * @brief Raw pointer to the first sorted index for each cell.
         */
        const int* cell_start_ptr {};

        /**
         * @brief Raw pointer to one-past-the-last sorted index for each cell.
         */
        const int* cell_end_ptr {};

        /**
         * @brief Number of particles reported by the fluid.
         */
        int particle_count {};

        /**
         * @brief Number of cells reported by the universe.
         *
         * @ref make_probe returns `true` only when this value is greater than zero.
         */
        int num_of_cells {};

        /**
         * @brief Volume of one universe cell.
         *
         * Copied from `universe->cell_volume()`.
         */
        T cell_volume {};

        /**
         * @brief Statistical weight of the fluid particles.
         *
         * Copied from `fluid->statistical_weight()`.
         */
        T statistical_weight {};
    };

public:
    /**
     * @brief Constructs an empty codec.
     *
     * All dependencies are initialized to null-equivalent values and the
     * solver-allocation buffer is empty. Calling @ref make_probe on an empty
     * codec returns `false`.
     */
    Codec() = default;

    /**
     * @brief Constructs a codec from required simulation dependencies.
     *
     * The constructor stores the provided dependencies and validates that all of
     * them are non-null. After validation, it calls @ref reset to resize
     * `d_allocated_solver` to the current universe cell count and fill it with
     * zero.
     *
     * @param domain Universe providing cell count, cell volume, and optional codec states.
     * @param fluid Fluid providing particle count and statistical weight.
     * @param searcher Spatial hashing searcher providing cell-to-particle ranges.
     *
     * @throw std::invalid_argument If `domain` is null.
     * @throw std::invalid_argument If `fluid` is null.
     * @throw std::invalid_argument If `searcher` is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Codec(UniverseHostPtr<T> domain,
          FluidHostPtr<T> fluid,
          SpatialHashingSearcherHostPtr<T> searcher);

    /**
     * @brief Destroys the codec through the base interface.
     */
    virtual ~Codec() = default;

    /**
     * @brief Executes one complete codec update cycle.
     *
     * The default implementation calls @ref encode first and @ref decode second.
     * Derived classes may override this function when their codec workflow
     * requires a different sequencing policy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update();

    /**
     * @brief Encodes the current simulation state into codec-owned or codec-target states.
     *
     * Derived classes implement this function to classify cells, compute
     * codec-specific quantities, or populate @ref d_allocated_solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode()
        = 0;

    /**
     * @brief Decodes codec-side data back into simulation state.
     *
     * Derived classes implement this function to apply results produced by the
     * codec workflow back to universe, fluid, or solver-facing state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode()
        = 0;

    /**
     * @brief Resizes and clears the per-cell solver-allocation buffer.
     *
     * The buffer is resized to `universe->number_of_cells()` and filled with
     * zero values. The constructor calls this after dependency validation.
     *
     * @warning This function assumes `_universe` is non-null. It should only be
     *          called on a codec whose universe dependency has been initialized.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Returns the mutable per-cell solver-allocation buffer.
     *
     * Derived codecs write this buffer to indicate which solver should be used
     * for each cell. The orchestrator exposes this buffer to codec-aware solvers
     * during solver dispatch.
     *
     * @return Mutable reference to the device allocation buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    allocated_solver() noexcept;

    /**
     * @brief Returns the immutable per-cell solver-allocation buffer.
     *
     * @return Const reference to the device allocation buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    allocated_solver() const noexcept;

    /**
     * @brief Replaces the per-cell fixed solver-index buffer.
     *
     * When the codec has a universe dependency, the replacement buffer must be
     * either empty or match the universe cell count.
     *
     * @param fixed_solver Per-cell solver indices for fixed-region cells.
     *
     * @throw std::invalid_argument If the buffer size does not match the universe cell count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fixed_solver(DeviceBuffer<int> fixed_solver);

    /**
     * @brief Returns the mutable per-cell fixed solver-index buffer.
     *
     * @return Mutable reference to the fixed solver-index buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    fixed_solver() noexcept;

    /**
     * @brief Returns the immutable per-cell fixed solver-index buffer.
     *
     * @return Const reference to the fixed solver-index buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    fixed_solver() const noexcept;

    /**
     * @brief Replaces the per-cell fixed-region mask.
     *
     * When the codec has a universe dependency, the replacement buffer must be
     * either empty or match the universe cell count. Cells with value `1` are
     * skipped by codec classification and decoded from @ref fixed_solver.
     *
     * @param fixed_region Per-cell fixed-region mask.
     *
     * @throw std::invalid_argument If the buffer size does not match the universe cell count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fixed_region(DeviceBuffer<int> fixed_region);

    /**
     * @brief Returns the mutable per-cell fixed-region mask.
     *
     * @return Mutable reference to the fixed-region mask.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    fixed_region() noexcept;

    /**
     * @brief Returns the immutable per-cell fixed-region mask.
     *
     * @return Const reference to the fixed-region mask.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    fixed_region() const noexcept;

    /**
     * @brief Refreshes the cached raw-pointer probe for derived codec kernels.
     *
     * This function resolves the configured universe, fluid, searcher, and
     * codec-owned allocation buffer into raw pointers and scalar metadata.
     *
     * The following pointers are optional and may be null:
     *
     * - `temperature_ptr`,
     * - `number_particle_ptr`,
     * - `knudsen_number_ptr`,
     * - `allocated_solver_ptr` when the allocation buffer is empty,
     * - `fixed_solver_ptr` when the fixed solver buffer is empty,
     * - `fixed_region_ptr` when the fixed-region buffer is empty,
     * - searcher pointers if the searcher does not currently expose buffers.
     *
     * The current success condition is intentionally lightweight:
     *
     * @code
     * return probe.num_of_cells > 0;
     * @endcode
     *
     * Therefore, derived codec implementations must explicitly validate any
     * pointer or state they require before use.
     *
     * @retval true Universe, fluid, and searcher dependencies exist, and the
     *              universe reports a positive cell count.
     * @retval false A required dependency is missing or the universe has no cells.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

protected:
    /**
     * @brief Universe dependency used for cell metadata and optional codec states.
     *
     * Required by the non-default constructor and @ref make_probe.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency used for particle metadata.
     *
     * Required by the non-default constructor and @ref make_probe.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency used for cell-to-particle ranges.
     *
     * Required by the non-default constructor and @ref make_probe.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Device buffer storing per-cell solver allocation metadata.
     *
     * The buffer is resized by @ref reset to match the universe cell count.
     * Derived codecs generally write solver identifiers or classification labels
     * into this buffer during @ref encode.
     */
    DeviceBuffer<int> d_allocated_solver;

    /**
     * @brief Device buffer storing solver indices for fixed-region cells.
     *
     * During decode, derived codecs should copy this value into
     * @ref d_allocated_solver for cells whose @ref d_fixed_region value is `1`.
     */
    DeviceBuffer<int> d_fixed_solver;

    /**
     * @brief Device buffer storing the per-cell fixed-region mask.
     *
     * A value of `1` marks cells that derived codecs should skip during encode
     * and decode from @ref d_fixed_solver.
     */
    DeviceBuffer<int> d_fixed_region;

    /**
     * @brief Cached probe populated by @ref make_probe.
     */
    CodecProbe _probe {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::Codec`.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using Codec = system::Codec<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::Codec`.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::Codec`.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

} // namespace atlas

#include <atlas/codec/codec.hpp>
