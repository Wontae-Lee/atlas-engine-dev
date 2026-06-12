#pragma once

/**
 * @file knudsen_codec.h
 * @brief Declares a Knudsen-number-based codec for cell-wise solver classification.
 *
 * @details
 * This file defines @ref atlas::KnudsenCodec, a codec specialization that
 * computes a cell-wise Knudsen number and maps the resulting scalar field to
 * discrete solver allocation indices.
 *
 * The codec is intended for hybrid, adaptive, or region-coupled simulations where
 * different numerical solvers are selected according to the local rarefaction
 * regime. A typical use case is a simulation domain where continuum-like,
 * transitional, and rarefied regions should be handled by different numerical
 * models.
 *
 * @section knudsen_codec_encoded_quantity Encoded quantity
 *
 * For each simulation cell, the codec estimates the local physical number density
 * from the simulation particle count:
 *
 * @f[
 *     n
 *     =
 *     \frac{N_p W}{V_{\mathrm{cell}}},
 * @f]
 *
 * where:
 *
 * - @f$N_p@f$ is the cell-wise simulation particle count,
 * - @f$W@f$ is the statistical particle weight,
 * - @f$V_{\mathrm{cell}}@f$ is the cell volume,
 * - @f$n@f$ is the resulting number density.
 *
 * With SI-consistent inputs, @f$n@f$ has units of @f$\mathrm{m}^{-3}@f$.
 *
 * A representative mean free path is then estimated using a hard-sphere-style
 * expression:
 *
 * @f[
 *     \lambda
 *     =
 *     \frac{1}{
 *         \sqrt{2}\,n\,\sigma
 *     },
 * @f]
 *
 * where:
 *
 * - @f$\lambda@f$ is the representative mean free path,
 * - @f$n@f$ is the number density,
 * - @f$\sigma@f$ is the representative collision cross-sectional area.
 *
 * With SI-consistent inputs, @f$\sigma@f$ should have units of
 * @f$\mathrm{m}^{2}@f$, and @f$\lambda@f$ has units of @f$\mathrm{m}@f$.
 *
 * The encoded Knudsen number is then:
 *
 * @f[
 *     \mathrm{Kn}
 *     =
 *     \frac{\lambda}{L},
 * @f]
 *
 * where @f$L@f$ is the characteristic length scale.
 *
 * Combining the above expressions gives:
 *
 * @f[
 *     \mathrm{Kn}
 *     =
 *     \frac{1}{
 *         \sqrt{2}\,n\,\sigma\,L
 *     }.
 * @f]
 *
 * @section knudsen_codec_solver_classification Solver classification
 *
 * The encoded Knudsen number is decoded into a solver index using split
 * thresholds. With the default thresholds:
 *
 * @f[
 *     \{0.01,\ 0.1,\ 1.0\},
 * @f]
 *
 * the mapping is:
 *
 * - @f$\mathrm{Kn} < 0.01@f$ maps to solver index `0`,
 * - @f$0.01 \le \mathrm{Kn} < 0.1@f$ maps to solver index `1`,
 * - @f$0.1 \le \mathrm{Kn} < 1.0@f$ maps to solver index `2`,
 * - @f$\mathrm{Kn} \ge 1.0@f$ maps to solver index `3`.
 *
 * @section knudsen_codec_fixed_regions Fixed regions
 *
 * The codec also supports fixed-region behavior. Cells marked by the fixed-region
 * mask bypass automatic Knudsen-based classification. During `encode()`, fixed
 * cells keep their previously stored Knudsen number. During `decode()`, fixed
 * cells receive prescribed solver indices from the fixed-solver buffer when that
 * buffer is available.
 *
 * @note
 * The mean-free-path model implemented by this codec uses a single global
 * representative collision cross-sectional area. It does not model
 * species-pair-dependent, velocity-dependent, or temperature-dependent collision
 * cross sections.
 *
 * @note
 * The default representative collision cross-sectional area is `T(1)`. This is
 * intended as a safe initialization value only. In SI-consistent production
 * simulations, users should provide a physically meaningful value in
 * @f$\mathrm{m}^{2}@f$ for the selected species or collision model.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas {

/**
 * @brief Codec for Knudsen-number-based cell classification.
 *
 * @details
 * `KnudsenCodec<T>` derives from `Codec<T>` and implements a two-stage
 * encode/decode workflow:
 *
 * 1. `encode()` computes a cell-wise Knudsen number.
 * 2. `decode()` maps the Knudsen number to a discrete solver allocation index.
 *
 * The encoded field is stored in `UniverseKnudsenNumberState<T>`. The decoded
 * solver index is written into the allocated-solver field managed by the shared
 * codec infrastructure.
 *
 * The codec estimates the number density in each cell from the particle-count
 * state, statistical particle weight, and cell volume. It then estimates a
 * representative mean free path using:
 *
 * @f[
 *     \lambda = \frac{1}{\sqrt{2}\,n\,\sigma}.
 * @f]
 *
 * The Knudsen number is then computed as:
 *
 * @f[
 *     \mathrm{Kn} = \frac{\lambda}{L}.
 * @f]
 *
 * This makes the final encoded value:
 *
 * @f[
 *     \mathrm{Kn}
 *     =
 *     \frac{1}{\sqrt{2}\,n\,\sigma\,L}.
 * @f]
 *
 * Fixed cells are excluded from automatic encoding and decoding. This allows
 * selected regions to remain assigned to a prescribed solver regardless of the
 * locally computed Knudsen number.
 *
 * @tparam T Floating-point scalar type used for codec computations.
 *
 * @note
 * The representative collision cross-sectional area is treated as a global model
 * parameter. Multi-species, species-pair-dependent, or temperature-dependent
 * collision models are not represented directly by this codec.
 *
 * @note
 * The computed Knudsen number is primarily intended for solver-region
 * classification. Its physical interpretation depends on whether the provided
 * characteristic length and representative collision cross-sectional area are
 * consistent with the target gas model.
 *
 * @see Codec
 */
template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    /**
     * @brief Builder for validated KnudsenCodec construction.
     *
     * @details
     * The builder stores required dependencies and optional fixed-region
     * configuration before constructing a `KnudsenCodec<T>`.
     *
     * Required configuration:
     *
     * - universe/domain object,
     * - fluid object,
     * - spatial hashing searcher,
     * - positive characteristic length,
     * - positive representative collision cross-sectional area.
     *
     * Optional configuration:
     *
     * - fixed solver index buffer,
     * - fixed-region mask buffer.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a default-initialized codec object.
     *
     * The resulting object does not contain the required universe, fluid, or
     * searcher dependencies. Therefore, a default-constructed codec is not ready
     * to run `encode()` or `decode()` unless the inherited codec state is
     * configured by another path.
     *
     * The scalar model parameters are initialized to safe defaults:
     *
     * - characteristic length: `T(1)`,
     * - representative collision cross-sectional area: `T(1)`.
     *
     * These defaults are not necessarily physically meaningful for production
     * simulations.
     */
    KnudsenCodec() = default;

    /**
     * @brief Constructs a Knudsen codec from the required runtime dependencies.
     *
     * @details
     * This constructor forwards the universe, fluid, and searcher to the base
     * `Codec<T>` class, stores the Knudsen model parameters, creates the universe
     * Knudsen-number state if necessary, and uploads the default split thresholds
     * to device memory.
     *
     * The encoded Knudsen number is:
     *
     * @f[
     *     \mathrm{Kn}
     *     =
     *     \frac{1}{
     *         \sqrt{2}\,n\,\sigma\,L
     *     },
     * @f]
     *
     * where:
     *
     * - @f$n@f$ is the local number density,
     * - @f$\sigma@f$ is the representative collision cross-sectional area,
     * - @f$L@f$ is the characteristic length.
     *
     * The default split thresholds are uploaded to `d_kn_split` and are used by
     * `decode()` to convert Knudsen numbers into solver indices.
     *
     * @param domain Universe object that provides cell topology and cell-wise
     *        states. The object must be non-null.
     * @param fluid Fluid object used by the shared codec infrastructure and for
     *        simulation-state access. The object must be non-null.
     * @param searcher Spatial hashing searcher required by the base `Codec<T>`
     *        interface. The object must be non-null.
     * @param characteristic_length Positive characteristic length @f$L@f$ used
     *        in @f$\mathrm{Kn}=\lambda/L@f$.
     * @param representative_collision_cross_sectional_area Positive representative
     *        collision cross-sectional area @f$\sigma@f$ used in
     *        @f$\lambda=1/(\sqrt{2}n\sigma)@f$.
     *
     * @throws std::invalid_argument Thrown if @p characteristic_length is not
     *         positive.
     * @throws std::invalid_argument Thrown if
     *         @p representative_collision_cross_sectional_area is not positive.
     *
     * @pre @p domain must be non-null.
     * @pre @p fluid must be non-null.
     * @pre @p searcher must be non-null.
     * @pre @p characteristic_length must be greater than zero.
     * @pre @p representative_collision_cross_sectional_area must be greater than zero.
     *
     * @post The universe contains `UniverseKnudsenNumberState<T>`.
     * @post The device-side split-threshold buffer contains the default thresholds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    KnudsenCodec(UniverseHostPtr<T> domain,
                 FluidHostPtr<T> fluid,
                 SpatialHashingSearcherHostPtr<T> searcher,
                 T characteristic_length,
                 T representative_collision_cross_sectional_area = T(1));

    /**
     * @brief Destructor.
     *
     * @details
     * Uses the default destruction behavior. Device buffers and shared pointers
     * are released according to their own RAII semantics.
     */
    ~KnudsenCodec() override = default;

    /**
     * @brief Computes the cell-wise Knudsen number.
     *
     * @details
     * For each non-fixed cell, this function computes:
     *
     * @f[
     *     n
     *     =
     *     \frac{N_p W}{V_{\mathrm{cell}}},
     *     \qquad
     *     \lambda
     *     =
     *     \frac{1}{\sqrt{2}n\sigma},
     *     \qquad
     *     \mathrm{Kn}
     *     =
     *     \frac{\lambda}{L}.
     * @f]
     *
     * The result is written into `UniverseKnudsenNumberState<T>`.
     *
     * The current implementation does not explicitly use temperature in the
     * mean-free-path estimate. The collision model is therefore a representative
     * hard-sphere-style model using a global cross-sectional area.
     *
     * Fixed-region cells are skipped and keep their previously stored Knudsen
     * values. This is useful when a region should be handled by a prescribed solver
     * regardless of the current density field.
     *
     * If a cell has invalid input data, for example non-positive number density,
     * the encoded Knudsen number is set to zero for that cell.
     *
     * @pre The codec must have access to `UniverseNumberParticleState<T>`.
     * @pre The codec must have access to `UniverseKnudsenNumberState<T>`.
     * @pre The characteristic length must be positive.
     * @pre The representative collision cross-sectional area must be positive.
     *
     * @post Non-fixed cells with valid density contain updated Knudsen numbers.
     * @post Non-fixed cells with invalid density contain zero.
     * @post Fixed cells retain their previously stored Knudsen numbers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode() override;

    /**
     * @brief Maps Knudsen numbers to solver allocation indices.
     *
     * @details
     * This function compares each cell-wise Knudsen number against the
     * split-threshold buffer and writes the resulting solver index into the
     * allocated-solver buffer.
     *
     * With the default thresholds @f$\{0.01, 0.1, 1.0\}@f$:
     *
     * @code
     * Kn < 0.01        -> solver 0
     * 0.01 <= Kn < 0.1 -> solver 1
     * 0.1  <= Kn < 1.0 -> solver 2
     * Kn >= 1.0        -> solver 3
     * @endcode
     *
     * More generally, if the split buffer contains @f$N@f$ thresholds, this
     * function may assign solver indices in the inclusive range @f$[0, N]@f$.
     *
     * Fixed cells bypass threshold-based classification. If a cell is marked as
     * fixed and a fixed-solver buffer is available, the corresponding fixed solver
     * value is copied directly into the allocated-solver buffer.
     *
     * @pre The codec must have access to `UniverseKnudsenNumberState<T>`.
     * @pre The codec must have access to the allocated-solver buffer.
     * @pre The split-threshold buffer must not be empty.
     *
     * @post Non-fixed cells receive threshold-based solver indices.
     * @post Fixed cells receive prescribed solver indices when a fixed-solver
     *       buffer is available.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode() override;

    /**
     * @brief Creates a builder for fluent KnudsenCodec construction.
     *
     * @details
     * The returned builder can be used to configure required dependencies,
     * Knudsen-model parameters, and optional fixed-region behavior before
     * constructing the codec.
     *
     * @return Fresh builder object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

private:
    /**
     * @brief Characteristic length used to normalize the mean free path.
     *
     * @details
     * This value corresponds to @f$L@f$ in:
     *
     * @f[
     *     \mathrm{Kn} = \frac{\lambda}{L}.
     * @f]
     *
     * It must be strictly positive for meaningful Knudsen-number computation.
     * In SI-consistent simulations, this value should be expressed in meters.
     *
     * The default value is `T(1)`, mainly to provide a safe initialization state.
     * Production simulations should set this value explicitly.
     */
    T _characteristic_length = T(1);

    /**
     * @brief Representative collision cross-sectional area.
     *
     * @details
     * This value corresponds to @f$\sigma@f$ in the representative mean-free-path
     * model:
     *
     * @f[
     *     \lambda
     *     =
     *     \frac{1}{\sqrt{2}n\sigma}.
     * @f]
     *
     * It must be strictly positive. In SI-consistent simulations, this value
     * should be expressed in @f$\mathrm{m}^{2}@f$.
     *
     * The default value is `T(1)`. This is intended only as a safe initialization
     * value. For physically meaningful simulations, this parameter should be set
     * to a species-specific, model-specific, or otherwise representative collision
     * cross-sectional area.
     */
    T _representative_collision_cross_sectional_area = T(1);

    /**
     * @brief Device-side Knudsen-number split thresholds.
     *
     * @details
     * This buffer stores monotonically increasing Knudsen-number thresholds used
     * by `decode()` to map a scalar Knudsen number to a solver index.
     *
     * If the buffer contains @f$N@f$ thresholds, the decoder may assign solver
     * indices in the range @f$[0, N]@f$.
     *
     * The default threshold set is:
     *
     * @f[
     *     \{0.01,\ 0.1,\ 1.0\}.
     * @f]
     *
     * The buffer is stored on the device because classification is performed
     * inside a device kernel.
     */
    DeviceBuffer<T> d_kn_split {};
};

/**
 * @brief Builder for KnudsenCodec.
 *
 * @details
 * The builder collects required dependencies, model parameters, and optional
 * fixed-region configuration before constructing a `KnudsenCodec<T>`.
 *
 * Required dependencies:
 *
 * - universe/domain,
 * - fluid,
 * - spatial hashing searcher,
 * - positive characteristic length,
 * - positive representative collision cross-sectional area.
 *
 * Optional configuration:
 *
 * - fixed solver buffer,
 * - fixed-region mask.
 *
 * The fixed-region mechanism allows selected cells to bypass automatic
 * Knudsen-based solver classification. This is useful for boundary regions,
 * embedded model regions, validation regions, or any cell group whose solver
 * assignment should be prescribed externally.
 *
 * @tparam T Floating-point scalar type used by the codec.
 */
template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty builder. Required dependencies are initially null.
     * Scalar model parameters are initialized to safe default values.
     */
    Builder() = default;

    /**
     * @brief Sets the universe/domain dependency.
     *
     * @details
     * The universe provides the cell layout and cell-wise states required by the
     * codec. The codec creates or writes `UniverseKnudsenNumberState<T>` through
     * this object and obtains the number of cells for fixed-buffer validation.
     *
     * @param domain Universe object providing cell layout and per-cell states.
     * @return Reference to this builder.
     *
     * @post The builder stores @p domain as the universe dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(UniverseHostPtr<T> domain) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @details
     * The fluid object is forwarded to the base codec infrastructure. It also
     * participates in probe construction, from which codec kernels access
     * simulation-state information such as statistical particle weight.
     *
     * @param fluid Fluid object used by the codec infrastructure.
     * @return Reference to this builder.
     *
     * @post The builder stores @p fluid as the fluid dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @details
     * The searcher is required by the shared `Codec<T>` construction interface.
     * It is forwarded to the base codec and kept as part of the codec runtime
     * dependencies.
     *
     * @param searcher Searcher required by the base `Codec<T>` interface.
     * @return Reference to this builder.
     *
     * @post The builder stores @p searcher as the searcher dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the characteristic length.
     *
     * @details
     * The characteristic length corresponds to @f$L@f$ in:
     *
     * @f[
     *     \mathrm{Kn} = \frac{\lambda}{L}.
     * @f]
     *
     * It must be strictly positive. Validation is deferred until `build()`,
     * `make_host_shared()`, or `validate()` is called.
     *
     * @param characteristic_length Positive characteristic length @f$L@f$.
     * @return Reference to this builder.
     *
     * @post The builder stores @p characteristic_length.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    /**
     * @brief Sets the representative collision cross-sectional area.
     *
     * @details
     * This value corresponds to @f$\sigma@f$ in:
     *
     * @f[
     *     \lambda
     *     =
     *     \frac{1}{\sqrt{2}n\sigma}.
     * @f]
     *
     * It is an area, not a length. In SI-consistent simulations, it should be
     * given in @f$\mathrm{m}^{2}@f$.
     *
     * Validation is deferred until `build()`, `make_host_shared()`, or
     * `validate()` is called.
     *
     * @param representative_collision_cross_sectional_area Positive representative
     *        collision cross-sectional area @f$\sigma@f$.
     * @return Reference to this builder.
     *
     * @post The builder stores @p representative_collision_cross_sectional_area.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_representative_collision_cross_sectional_area(
        T representative_collision_cross_sectional_area) noexcept;

    /**
     * @brief Sets prescribed solver indices for fixed cells.
     *
     * @details
     * The fixed-solver buffer stores per-cell solver indices that can override
     * automatic threshold-based solver classification.
     *
     * During `decode()`, if a cell is marked as fixed and this buffer is
     * available, the corresponding value from this buffer is copied into the
     * allocated-solver buffer.
     *
     * The buffer may be empty. If it is non-empty, its size must match the universe
     * cell count.
     *
     * @param fixed_solver Per-cell solver indices for fixed cells.
     * @return Reference to this builder.
     *
     * @pre If non-empty, @p fixed_solver must contain one entry per universe cell.
     * @post The builder stores @p fixed_solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    /**
     * @brief Sets the fixed-region mask.
     *
     * @details
     * The fixed-region mask identifies cells that should bypass automatic
     * Knudsen-based classification.
     *
     * A cell with value `1` is treated as fixed:
     *
     * - `encode()` skips the cell and leaves its Knudsen number unchanged.
     * - `decode()` copies the prescribed solver index from the fixed-solver buffer
     *   when that buffer is available.
     *
     * A cell with any value other than `1` is treated as non-fixed and follows the
     * normal encode/decode path.
     *
     * The buffer may be empty. If it is non-empty, its size must match the universe
     * cell count.
     *
     * @param fixed_region Per-cell fixed-region flags.
     * @return Reference to this builder.
     *
     * @pre If non-empty, @p fixed_region must contain one entry per universe cell.
     * @post The builder stores @p fixed_region.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    /**
     * @brief Validates the builder state and constructs a codec by value.
     *
     * @details
     * This function validates all required dependencies, scalar model parameters,
     * and optional fixed buffers. It then constructs a `KnudsenCodec<T>` and
     * attaches optional fixed-solver and fixed-region buffers when they are
     * provided.
     *
     * @return Constructed `KnudsenCodec<T>` object.
     *
     * @throws std::invalid_argument Thrown if:
     * - the universe pointer is null,
     * - the fluid pointer is null,
     * - the searcher pointer is null,
     * - the characteristic length is not positive,
     * - the representative collision cross-sectional area is not positive,
     * - the fixed-solver buffer is non-empty and does not match the universe cell count,
     * - the fixed-region buffer is non-empty and does not match the universe cell count.
     *
     * @post The returned codec contains the configured model parameters.
     * @post Optional fixed buffers are attached to the returned codec when provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    /**
     * @brief Validates the builder state and constructs a host-shared codec.
     *
     * @details
     * This function behaves like `build()`, but allocates the codec as an
     * `atlas::host_shared_ptr`. Use this function when the codec needs shared
     * ownership on the host side.
     *
     * @return Host shared pointer to the constructed codec.
     *
     * @throws std::invalid_argument Thrown if:
     * - the universe pointer is null,
     * - the fluid pointer is null,
     * - the searcher pointer is null,
     * - the characteristic length is not positive,
     * - the representative collision cross-sectional area is not positive,
     * - the fixed-solver buffer is non-empty and does not match the universe cell count,
     * - the fixed-region buffer is non-empty and does not match the universe cell count.
     *
     * @post The returned codec contains the configured model parameters.
     * @post Optional fixed buffers are attached to the returned codec when provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates required dependencies, model parameters, and fixed buffers.
     *
     * @details
     * This function checks whether the builder contains all dependencies required
     * to construct a valid codec. It also checks that all scalar model parameters
     * are positive and that optional per-cell buffers have sizes compatible with
     * the universe cell count.
     *
     * @throws std::invalid_argument Thrown if:
     * - the universe pointer is null,
     * - the fluid pointer is null,
     * - the searcher pointer is null,
     * - the characteristic length is not positive,
     * - the representative collision cross-sectional area is not positive,
     * - `fixed_solver` is non-empty and does not match the universe cell count,
     * - `fixed_region` is non-empty and does not match the universe cell count.
     *
     * @pre None.
     * @post No builder state is modified.
     */
    void
    validate() const;

private:
    /**
     * @brief Universe/domain dependency.
     *
     * @details
     * The universe defines the cell topology and stores cell-wise states required
     * by encoding and decoding.
     */
    UniverseHostPtr<T> _domain {};

    /**
     * @brief Fluid dependency.
     *
     * @details
     * The fluid object is forwarded to the base codec infrastructure and
     * participates in probe construction.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial searcher dependency.
     *
     * @details
     * The searcher is required by the base `Codec<T>` interface.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Characteristic length @f$L@f$.
     *
     * @details
     * This value normalizes the representative mean free path into a Knudsen
     * number. It must be positive.
     */
    T _characteristic_length = T(1);

    /**
     * @brief Representative collision cross-sectional area.
     *
     * @details
     * This value corresponds to @f$\sigma@f$ in:
     *
     * @f[
     *     \lambda
     *     =
     *     \frac{1}{\sqrt{2}n\sigma}.
     * @f]
     *
     * In SI-consistent simulations, this value should be given in
     * @f$\mathrm{m}^{2}@f$.
     *
     * The default value is intended only as a safe initialization value.
     * Production simulations should provide a physically meaningful
     * species-specific or model-specific value.
     */
    T _representative_collision_cross_sectional_area = T(1);

    /**
     * @brief Optional fixed solver index buffer.
     *
     * @details
     * Stores prescribed per-cell solver indices for fixed cells. If non-empty,
     * the buffer size must match the universe cell count.
     */
    DeviceBuffer<int> _fixed_solver {};

    /**
     * @brief Optional fixed-region mask buffer.
     *
     * @details
     * Stores per-cell flags indicating which cells bypass automatic
     * Knudsen-based classification. A value of `1` marks a fixed cell.
     * If non-empty, the buffer size must match the universe cell count.
     */
    DeviceBuffer<int> _fixed_region {};
};

} // namespace atlas

namespace atlas {


/**
 * @brief Host shared pointer alias for @ref atlas::KnudsenCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec<T>>;

/**
 * @brief Device shared pointer alias for @ref atlas::KnudsenCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec<T>>;

} // namespace atlas

#include <atlas/codec/knudsen_codec.hpp>