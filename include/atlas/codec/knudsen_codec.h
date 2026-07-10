#pragma once

#include <atlas/container/container.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>

namespace atlas {

/**
 * @brief Codec leaf that selects a per-cell solver from the cell's Knudsen number.
 *
 * For every grid cell, `allocate()` converts the cell's particle count into a
 * rarefaction measure (the Knudsen number) and buckets it against a fixed set of
 * split thresholds to yield an integer solver index, which `System::solve()`
 * later dispatches on. Small Knudsen numbers (dense, continuum-like flow) map to
 * the low indices; large ones (rarefied, free-molecular flow) map to the high
 * indices.
 *
 * The four inputs to the Knudsen number are held as **representative scalars**
 * standing in for the whole domain — a single characteristic length, collision
 * cross-section, statistical weight, and cell volume — rather than per-cell
 * fields. This is deliberate: the engine favors a fast, domain-wide estimate over
 * fine per-cell physical detail.
 *
 * The type is **trivially copyable** (all members are `float` or a fixed-size
 * `Container`, never a `DeviceBuffer`) so that a whole `KnudsenCodec` can be
 * captured by value inside the device lambda in `allocate()` and evaluated on the
 * GPU. `knudsen_number()` and `solver_index()` are therefore `__host__ __device__`.
 *
 * @see Codec, CodecType, UniverseAllocatedSolverState
 */
class KnudsenCodec final {
public:
    /// Builder that validates the four representative scalars, then constructs a leaf.
    class Builder;

    /// Number of split thresholds; a table of N splits produces N+1 solver indices (here 5).
    static constexpr int split_count = 4;

    /// Fixed-size, trivially-copyable table of ascending Knudsen thresholds.
    using SplitTable = Container<float, static_cast<std::size_t>(split_count)>;

public:
    /// Default-construct with all representative scalars at 1.0 and the default split table.
    KnudsenCodec() = default;

    /**
     * @brief Construct from the four representative scalars; the split table stays fixed.
     *
     * No validation is performed here — the `Builder` is the validating entry point.
     * Constructing directly with a non-positive scalar yields a codec whose
     * `knudsen_number()` guards evaluate to 0 for those cells (see that function).
     *
     * @param representative_characteristic_length   Domain length scale [m], divides
     *        the mean free path; must be positive to produce a nonzero Knudsen number.
     * @param representative_collision_cross_sectional_area Collision cross-section
     *        [m^2] used in the mean-free-path denominator; positive for a nonzero result.
     * @param representative_statistical_weight       Real particles represented by one
     *        simulated particle; scales the particle count into a number density.
     * @param representative_cell_volume              Cell volume [m^3] dividing the
     *        weighted particle count into a number density; positive to avoid a no-op.
     */
    ATLAS_HOST
    KnudsenCodec(float representative_characteristic_length,
                 float representative_collision_cross_sectional_area,
                 float representative_statistical_weight,
                 float representative_cell_volume);

    /**
     * @brief Create a fresh `Builder` for fluent configuration.
     * @return A default-initialized builder (all scalars 1.0).
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Compute and store a solver index for every cell in the universe.
     *
     * Launches one device thread per cell: each reads the cell's particle count
     * from @p number_particle, evaluates `solver_index(knudsen_number(count))`,
     * and writes the result into @p allocated_solver at the same cell index.
     *
     * The call is a **no-op** when @p number_particle or @p allocated_solver is
     * null, when the allocated-solver buffer is empty, or when the two buffers'
     * sizes disagree — the codec never resizes the output.
     *
     * @param temperature   Per-cell temperature state; accepted for the shared
     *        codec-operation shape but **unused** by this leaf, so it may be null.
     * @param number_particle Per-cell particle counts (input); read on the device.
     * @param allocated_solver Per-cell solver indices (output); overwritten in place.
     * @note Runs entirely on the device; there is no host round-trip of the data.
     */
    ATLAS_HOST void
    allocate(const UniverseTemperatureState* temperature,
             const UniverseNumberParticleState* number_particle,
             UniverseAllocatedSolverState* allocated_solver) const;

    /**
     * @brief Estimate a cell's Knudsen number from its simulated-particle count.
     *
     * Number density is `particle_count * statistical_weight / cell_volume`; the
     * mean free path is `1 / (sqrt(2) * n * cross_section)`; the Knudsen number is
     * that mean free path divided by the characteristic length. Callable on host
     * and device so the device lambda in `allocate()` can invoke it directly.
     *
     * Returns 0 (a deliberate, safe fallback rather than NaN/inf) whenever the cell
     * volume, resulting number density, characteristic length, or cross-section is
     * non-positive — i.e. whenever the formula would divide by zero or is
     * physically meaningless. The `!(x > 0)` guards also reject NaN inputs.
     *
     * @param particle_count Simulated particles in the cell (may be 0, giving Kn 0).
     * @return The dimensionless Knudsen number, or 0 when it cannot be formed.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    knudsen_number(const float particle_count) const noexcept {
        const float number_density = _representative_cell_volume > 0.0f
            ? particle_count * _representative_statistical_weight / _representative_cell_volume
            : 0.0f;

        if (!(number_density > 0.0f)
            || !(_representative_characteristic_length > 0.0f)
            || !(_representative_collision_cross_sectional_area > 0.0f)) {
            return 0.0f;
        }

        const float mean_free_path = 1.0f
            / (atlas::SQRT_TWO
               * number_density
               * _representative_collision_cross_sectional_area);
        return mean_free_path / _representative_characteristic_length;
    }

    /**
     * @brief Bucket a Knudsen number into a solver index against the split table.
     *
     * Walks the ascending split thresholds and returns the count of thresholds the
     * value meets or exceeds: the result is in `[0, split_count]`, i.e. one of five
     * indices for the default four-entry table. A `kn` below the first split maps to
     * 0; a `kn` at or above the last split maps to `split_count`. The `!(kn < split)`
     * test (rather than `kn >= split`) makes a NaN `kn` fall through to index 0.
     * Callable on host and device.
     *
     * @param kn A Knudsen number, typically from `knudsen_number()`.
     * @return The chosen solver index in the range `[0, split_count]`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    solver_index(const float kn) const noexcept {
        int index = 0;
        while (index < split_count && !(kn < _kn_split[static_cast<std::size_t>(index)])) {
            ++index;
        }
        return index;
    }

private:
    /// Domain length scale [m] dividing the mean free path; positive for a nonzero Kn.
    float _representative_characteristic_length = 1.0f;

    /// Collision cross-section [m^2] in the mean-free-path denominator.
    float _representative_collision_cross_sectional_area = 1.0f;

    /// Real particles per simulated particle; scales the count into a number density.
    float _representative_statistical_weight = 1.0f;

    /// Cell volume [m^3] dividing the weighted count into a number density.
    float _representative_cell_volume = 1.0f;

    /// Ascending Knudsen thresholds; fixed at construction and not builder-configurable.
    SplitTable _kn_split { 0.01f, 0.1f, 1.0f, 10.0f };
};

/**
 * @brief Fluent, validating builder for `KnudsenCodec`.
 *
 * Each `with_*` setter overrides one representative scalar and returns `*this` for
 * chaining; the split table is not configurable. `build()` runs `validate()` first,
 * which throws if any scalar is non-positive, so a successfully built codec always
 * has a well-formed set of representative values.
 */
class KnudsenCodec::Builder final {
public:
    /// Start with every representative scalar at 1.0.
    Builder() = default;

    /**
     * @brief Set the representative characteristic length [m].
     * @param representative_characteristic_length Must be positive (checked in `validate()`).
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_representative_characteristic_length(float representative_characteristic_length) noexcept;

    /**
     * @brief Set the representative collision cross-sectional area [m^2].
     * @param representative_collision_cross_sectional_area Must be positive (checked in `validate()`).
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_representative_collision_cross_sectional_area(
        float representative_collision_cross_sectional_area) noexcept;

    /**
     * @brief Set the representative statistical weight (real particles per simulated particle).
     * @param representative_statistical_weight Must be positive (checked in `validate()`).
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_representative_statistical_weight(float representative_statistical_weight) noexcept;

    /**
     * @brief Set the representative cell volume [m^3].
     * @param representative_cell_volume Must be positive (checked in `validate()`).
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_representative_cell_volume(float representative_cell_volume) noexcept;

    /**
     * @brief Validate the accumulated scalars, then construct the codec by value.
     * @return A fully configured `KnudsenCodec`.
     * @throws std::invalid_argument if any representative scalar is not positive.
     */
    ATLAS_NODISCARD ATLAS_HOST KnudsenCodec
    build() const;

    /**
     * @brief Build and wrap the codec in a host-side shared pointer.
     * @return A `host_shared_ptr` owning a freshly built `KnudsenCodec`.
     * @throws std::invalid_argument if `build()`'s validation fails.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<KnudsenCodec>
    make_host_shared() const;

private:
    /**
     * @brief Throw if any accumulated representative scalar is non-positive.
     * @throws std::invalid_argument naming the first offending scalar.
     */
    ATLAS_HOST void
    validate() const;

private:
    /// Pending characteristic length [m]; copied into the codec by `build()`.
    float _representative_characteristic_length = 1.0f;

    /// Pending collision cross-sectional area [m^2]; copied into the codec by `build()`.
    float _representative_collision_cross_sectional_area = 1.0f;

    /// Pending statistical weight; copied into the codec by `build()`.
    float _representative_statistical_weight = 1.0f;

    /// Pending cell volume [m^3]; copied into the codec by `build()`.
    float _representative_cell_volume = 1.0f;
};

/// Host-side shared owner of a `KnudsenCodec` leaf.
using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec>;

/// Device-side shared owner of a `KnudsenCodec` leaf.
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec>;

}