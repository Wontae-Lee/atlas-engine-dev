#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

/**
 * @file knudsen_codec.h
 * @brief Classifies each cell by local Knudsen number into a discrete
 *        flow-regime index, the standard criterion for deciding where a
 *        continuum (Navier-Stokes/SPH-like) approximation is valid
 *        versus where particle methods (DSMC) are required.
 *
 * @details
 * ### Background
 * The Knudsen number `Kn = lambda / L` (mean free path `lambda` over a
 * macroscopic characteristic length `L`) measures how "rarefied" a gas
 * is relative to the length scale of interest. It is the standard
 * criterion in rarefied gas dynamics for choosing a simulation method:
 * - `Kn < 0.01`: continuum regime — the Navier-Stokes equations (and
 *   continuum solvers like `SphSolver`) are valid.
 * - `0.01 <= Kn < 0.1`: slip flow — continuum equations still apply away
 *   from walls, with modified (slip) boundary conditions.
 * - `0.1 <= Kn < 1`: transitional regime — neither continuum equations
 *   nor free-molecular approximations are accurate; particle methods
 *   (DSMC) are needed.
 * - `Kn >= 1`: free-molecular regime — intermolecular collisions are
 *   rare enough that particles mostly interact only with boundaries.
 * This is the classification this codec implements: `d_kn_split`
 * defaults to `{0.01, 0.1, 1.0}` (this book-standard threshold set,
 * following e.g. Bird 1994 / Karniadakis, Beskok & Aluru 2005), mapping
 * a cell's Knudsen number to a solver index `0`-`3` over those four
 * regimes.
 *
 * ### Derivation — mean free path and Knudsen number
 * For a hard-sphere gas, the mean free path between collisions is
 * `lambda = 1 / (sqrt(2) * n * sigma)`, where `n` is the number density
 * and `sigma` the collision cross-section; the `sqrt(2)` factor accounts
 * for the fact that both colliding molecules are themselves moving
 * (relative-speed-averaged over the Maxwell-Boltzmann distribution),
 * rather than one being a stationary target — a standard kinetic-theory
 * result (see G. A. Bird, "Molecular Gas Dynamics and the Direct
 * Simulation of Gas Flows," 1994, or any statistical-mechanics text on
 * mean free path). `knudsen_number()` computes `n = particle_count *
 * statistical_weight / cell_volume` (real molecules per unit volume,
 * from the simulated particle count scaled by the DSMC statistical
 * weight — see `dsmc_solver.h`) and returns `lambda / characteristic_length`.
 *
 * ### Operating principle
 * `encode()` (`encode_cells`) computes and stores each non-fixed cell's
 * Knudsen number into `UniverseKnudsenNumberState`. `decode()`
 * (`decode_cells`) maps that value through `solver_index()` — a linear
 * scan finding the first split threshold the Knudsen number is below —
 * into `Codec::allocated_solver()`, or substitutes `fixed_solver_ptr`
 * directly for cells flagged by `fixed_region_ptr` (see `codec.h`'s
 * fixed-region override mechanism).
 *
 * ### References
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994. (mean free path, Knudsen
 *   number, and flow-regime classification)
 * - G. Karniadakis, A. Beskok, and N. Aluru, "Microflows and
 *   Nanoflows: Fundamentals and Simulation," Springer, 2005. (the
 *   commonly cited `{0.01, 0.1, 1}` regime split thresholds)
 */

namespace atlas {

/**
 * @brief Knudsen-number-based per-cell solver classifier. See this
 *        file's top-of-file documentation for the flow-regime
 *        thresholds and the mean-free-path derivation.
 */
class KnudsenCodec final : public Codec {
public:
    class Builder;

    KnudsenCodec() = default;

    ATLAS_HOST KnudsenCodec(UniverseHostPtr domain,
                            FluidHostPtr fluid,
                            SearcherHostPtr searcher,
                            float characteristic_length,
                            float representative_collision_cross_sectional_area = 1.0f);

    ~KnudsenCodec() override = default;

    /** @brief Computes and stores each non-fixed cell's Knudsen number
     *  (`knudsen_number()`) into `UniverseKnudsenNumberState`. No-op if
     *  the probe can't be built or the required states don't exist. */
    ATLAS_HOST void
    encode() override;

    /** @brief Maps each non-fixed cell's stored Knudsen number to a
     *  solver index (`solver_index()`) into `Codec::allocated_solver()`;
     *  fixed cells take `fixed_solver_ptr[cell]` directly. No-op if the
     *  probe can't be built or the required states/splits don't exist. */
    ATLAS_HOST void
    decode() override;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

public:
    ATLAS_HOST void
    encode_cells();

    ATLAS_HOST void
    decode_cells();

private:
    /** @brief Whether `cell` is exempt from automatic classification
     *  (`CodecProbe::fixed_region_ptr[cell] == 1`); see `codec.h`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    fixed_cell(const CodecProbe& probe, const int cell) noexcept {
        return probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1;
    }

    /**
     * @brief `Kn = lambda / characteristic_length`,
     *        `lambda = 1 / (sqrt(2) * n * sigma)` (hard-sphere mean free
     *        path), `n = particle_count * statistical_weight /
     *        cell_volume`, `sigma =
     *        representative_collision_cross_sectional_area`. See this
     *        file's top-of-file Derivation. Returns `0` if any input
     *        (density, characteristic length, cross-section) is
     *        non-positive.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    knudsen_number(const float particle_count,
                   const float statistical_weight,
                   const float cell_volume,
                   const float characteristic_length,
                   const float representative_collision_cross_sectional_area) noexcept {
        const float number_density = cell_volume > 0.0f
            ? particle_count * statistical_weight / cell_volume
            : 0.0f;

        if (!(number_density > 0.0f)
            || !(characteristic_length > 0.0f)
            || !(representative_collision_cross_sectional_area > 0.0f)) {
            return 0.0f;
        }

        const float mean_free_path = 1.0f
            / (atlas::SQRT_TWO
               * number_density
               * representative_collision_cross_sectional_area);
        return mean_free_path / characteristic_length;
    }

    /**
     * @brief Linear search for the first `splits[i]` that `kn` is
     *        strictly below; returns that `i`, or `split_count` if `kn`
     *        exceeds every split. With the default splits `{0.01, 0.1,
     *        1.0}` this yields `0` (continuum), `1` (slip), `2`
     *        (transitional), `3` (free-molecular) — see this file's
     *        top-of-file documentation for the regime meanings.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    solver_index(const float kn, const float* splits, const int split_count) noexcept {
        int index = 0;
        while (index < split_count && !(kn < splits[index])) {
            ++index;
        }
        return index;
    }

private:
    float _characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    DeviceBuffer<float> d_kn_split {};
};

/**
 * @brief Fluent builder for `KnudsenCodec`. Validation requires
 *        non-null `_domain`/`_fluid`/`_searcher`, positive
 *        `_characteristic_length`/`_representative_collision_cross_sectional_area`,
 *        and `_fixed_solver`/`_fixed_region` either empty or sized to
 *        the domain's cell count.
 */
class KnudsenCodec::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_domain(UniverseHostPtr domain) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_characteristic_length(float characteristic_length) noexcept;

    ATLAS_HOST Builder&
    with_representative_collision_cross_sectional_area(
        float representative_collision_cross_sectional_area) noexcept;

    ATLAS_HOST Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    ATLAS_HOST Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    ATLAS_HOST ATLAS_NODISCARD KnudsenCodec
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<KnudsenCodec>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _domain {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    float _characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    DeviceBuffer<int> _fixed_solver {};

    DeviceBuffer<int> _fixed_region {};
};

using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec>;

using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec>;

}
