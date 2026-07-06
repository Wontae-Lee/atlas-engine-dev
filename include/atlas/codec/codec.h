#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

/**
 * @file codec.h
 * @brief Host-only base interface for per-cell flow-regime
 *        classification: produces the `allocated_solver` buffer that
 *        `Solver::solve(allocated_solver, index, dt)` uses to restrict
 *        several solvers to disjoint cell subsets of one universe.
 *
 * @details
 * ### Background
 * `solver.h` documents the *consumer* side of per-cell multi-solver
 * allocation: a `Solver` restricted by `allocated_solver[cell] == index`.
 * `Codec` is the *producer*: it inspects each cell's local flow state
 * (density, temperature, ...) and decides which solver index that cell
 * should be handled by this step, writing the result into
 * `d_allocated_solver`. This is a *cell-level*, precomputed-map approach
 * to hybrid solver coupling: a `Codec`-driven setup runs several
 * independent `Solver` instances (e.g. one `DsmcSolver`, one
 * `SphSolver`) each restricted to the cells the codec currently assigns
 * them.
 *
 * `encode()`/`decode()` are named after the general encode/decode
 * (compress/reconstruct) framing: `encode()` distills raw per-cell flow
 * state into a compact per-cell *code* (e.g. `KnudsenCodec`'s
 * `knudsen_number_ptr`), and `decode()` maps that code to the concrete
 * *decision* (`allocated_solver_ptr`, a solver index per cell) —
 * `update()` runs both in sequence each time the classification should
 * be refreshed.
 *
 * ### Operating principle
 * `d_allocated_solver`/`d_fixed_solver`/`d_fixed_region` are all
 * per-cell `DeviceBuffer<int>`s sized to `Universe::cell_count()`
 * (`reset()`, called from the constructor). `fixed_region`/`fixed_solver`
 * are an escape hatch: a cell with `fixed_region[cell] == 1` is exempt
 * from whatever automatic classification rule a concrete `Codec` applies
 * and instead always resolves to `fixed_solver[cell]` — letting a caller
 * pin specific cells (e.g. a known inlet or a diagnostic region) to a
 * chosen solver regardless of the codec's own criterion. `make_probe()`
 * assembles the `CodecProbe` view (`detail::CodecProbeBuilder::make`)
 * concrete codecs' `encode()`/`decode()` operate through.
 */

namespace atlas {

/**
 * @brief Base interface for per-cell solver-allocation classifiers. See
 *        this file's top-of-file documentation for the encode/decode
 *        framing and the fixed-region override mechanism.
 */
class Codec {
public:
    Codec() = default;

    ATLAS_HOST
    Codec(UniverseHostPtr domain,
          FluidHostPtr fluid,
          SearcherHostPtr searcher);

    virtual ~Codec() = default;

    Codec(const Codec&) = default;
    Codec&
    operator=(const Codec&)
        = default;
    Codec(Codec&&) noexcept = default;
    Codec&
    operator=(Codec&&) noexcept = default;

    /** @brief `encode()` then `decode()` — refreshes
     *  `allocated_solver()` from the current flow state in one call. */
    ATLAS_HOST virtual void
    update();

    /** @brief Distills raw per-cell flow state into whatever
     *  intermediate per-cell "code" the concrete codec uses (e.g.
     *  `KnudsenCodec`'s per-cell Knudsen number). */
    ATLAS_HOST virtual void
    encode()
        = 0;

    /** @brief Maps the code produced by `encode()` to a concrete
     *  per-cell solver index, writing `allocated_solver()`. */
    ATLAS_HOST virtual void
    decode()
        = 0;

    /** @brief Resizes `allocated_solver()`/`fixed_solver()`/
     *  `fixed_region()` to `Universe::cell_count()`, zero-filled. */
    ATLAS_HOST void
    reset() noexcept;

    /** @brief Per-cell solver index this codec last decoded; the buffer
     *  `Solver::solve(allocated_solver, index, dt)` is called with. */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    allocated_solver() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    allocated_solver() const noexcept;

    /** @brief Sets the per-cell solver-index override used wherever
     *  `fixed_region()[cell] == 1`. Must be empty or sized to
     *  `Universe::cell_count()`. */
    ATLAS_HOST void
    set_fixed_solver(DeviceBuffer<int> fixed_solver);

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    fixed_solver() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    fixed_solver() const noexcept;

    /** @brief Sets the per-cell "exempt from automatic classification"
     *  flags (`1` = pinned to `fixed_solver()[cell]`). Must be empty or
     *  sized to `Universe::cell_count()`. */
    ATLAS_HOST void
    set_fixed_region(DeviceBuffer<int> fixed_region);

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    fixed_region() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    fixed_region() const noexcept;

    /** @brief Rebuilds `_probe` from the current universe/fluid/searcher
     *  state and this codec's allocation buffers
     *  (`detail::CodecProbeBuilder::make`); `false` if
     *  universe/fluid/searcher is null or the universe has zero cells. */
    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    DeviceBuffer<int> d_allocated_solver;

    DeviceBuffer<int> d_fixed_solver;

    DeviceBuffer<int> d_fixed_region;

    CodecProbe _probe {};
};

using CodecHostPtr = atlas::host_shared_ptr<Codec>;

using CodecDevicePtr = atlas::device_shared_ptr<Codec>;

}
