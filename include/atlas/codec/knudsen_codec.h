#pragma once

/**
 * @file knudsen_codec.h
 * @brief Declares a codec that evaluates and stores a Knudsen-related field over the simulation domain.
 *
 * @details
 * This header defines @ref atlas::system::KnudsenCodec, a concrete
 * @ref atlas::system::Codec implementation intended for workflows in which
 * simulation state is reduced to, or interpreted through, a Knudsen-style field.
 *
 * In continuum-to-rarefied flow settings, the Knudsen number is commonly used as
 * a nondimensional indicator relating a microscopic length scale to a macroscopic
 * reference scale. In practical simulation software, a Knudsen-like quantity may
 * be used to:
 * - classify cells or regions by flow regime,
 * - trigger model switching or hybrid solver behavior,
 * - guide refinement, stabilization, or reconstruction logic,
 * - expose a compact diagnostic field derived from particle and grid state.
 *
 * This codec augments the generic @ref atlas::system::Codec interface with:
 * - a characteristic reference length,
 * - device-resident storage for per-cell Knudsen-derived values,
 * - encode/decode hooks that can populate and consume that reduced field.
 *
 * ## Intended role
 * A typical use pattern is:
 * 1. Associate the codec with a simulation @ref atlas::domain::Domain.
 * 2. Provide a characteristic length scale relevant to the problem setup.
 * 3. Call @ref encode to evaluate or refresh a per-cell Knudsen field.
 * 4. Optionally call @ref decode to project or consume that field in downstream logic.
 *
 * ## Representation
 * The exact definition of the stored Knudsen-related quantity is implementation-defined.
 * Depending on the implementation in `knudsen_codec.hpp`, the stored values may represent:
 * - a classical Knudsen number,
 * - a proxy or surrogate rarefaction metric,
 * - a regime-classification scalar derived from local state.
 *
 * ## Construction
 * The codec may be:
 * - default-constructed, or
 * - created through the nested @ref Builder, which stages the domain and
 *   characteristic length before validation and construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout simulation and codec operations.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Concrete codec that evaluates a Knudsen-style field over the simulation domain.
 *
 * @details
 * @ref KnudsenCodec is a specialization of @ref Codec that computes and stores
 * a per-cell scalar field associated with Knudsen-style rarefaction analysis or
 * related nondimensional diagnostics.
 *
 * Relative to the abstract codec interface, this class introduces:
 * - a characteristic reference length used to scale or interpret local quantities,
 * - a device buffer holding per-cell Knudsen-derived values,
 * - encode/decode hooks specialized for this field representation.
 *
 * ## Conceptual interpretation
 * In many applications, a Knudsen-like value relates:
 * - a local microscopic or mean-free-path-like measure,
 * - to a macroscopic characteristic length scale.
 *
 * Although the exact formulation is implementation-defined, the codec is intended
 * to support logic such as:
 * - regime detection,
 * - solver selection,
 * - postprocessing diagnostics,
 * - reduced-field transport between simulation stages.
 *
 * ## Domain association
 * The codec is typically associated with a @ref Domain so it can evaluate and store
 * values aligned with the simulation grid or persistent field layout.
 *
 * ## Runtime type
 * This class reports @ref CodecType::knudsen from @ref type().
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for @ref KnudsenCodec.
     *
     * @details
     * The builder stages required construction inputs such as the simulation domain
     * and characteristic length, validates them, and then materializes either:
     * - a value instance of @ref KnudsenCodec, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a codec with default-initialized state, including a unit
     * characteristic length.
     *
     * @note
     * A default-constructed codec may still require a valid domain and a
     * problem-appropriate characteristic length before meaningful use.
     */
    KnudsenCodec() = default;

    /**
     * @brief Construct the codec with a domain and reference length scale.
     *
     * @details
     * Associates the codec with the supplied simulation domain and stores the
     * characteristic length used when evaluating Knudsen-derived quantities.
     *
     * The characteristic length commonly represents a macroscopic problem scale,
     * geometric reference scale, or local-model normalization factor, depending
     * on the implementation.
     *
     * @param domain Host-side shared pointer to the associated simulation domain.
     * @param characteristic_length Reference length used in Knudsen evaluation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length);

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Defaulted override for correct destruction through base-class pointers.
     */
    ~KnudsenCodec() override = default;

    /**
     * @brief Encode simulation state into a per-cell Knudsen-derived field.
     *
     * @details
     * This function is the primary forward transformation entry point for the
     * Knudsen codec.
     *
     * In a typical implementation, it may:
     * - inspect particle-resolved state,
     * - access domain-aligned fields and metadata,
     * - use neighborhood information from the spatial searcher,
     * - evaluate a local Knudsen-related scalar for each cell,
     * - store the resulting field into codec-managed device memory.
     *
     * The exact evaluation rule is implementation-defined in `knudsen_codec.hpp`.
     *
     * @param particle_probe Probe exposing the active fluid/particle state.
     * @param domain_probe Probe exposing domain field data and grid metadata.
     * @param searcher_probe Probe exposing spatial neighborhood indexing state.
     * @param codec_probe Mutable codec-side probe carrying codec-specific device state.
     *
     * @note
     * Depending on the implementation, this function may lazily initialize or
     * resize internal device storage before writing the Knudsen field.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decode the Knudsen-derived field into downstream runtime state.
     *
     * @details
     * This function is the inverse-facing entry point for the Knudsen codec.
     *
     * In a typical implementation, it may:
     * - read previously encoded per-cell Knudsen values,
     * - classify cells or particles by regime,
     * - reconstruct auxiliary state needed by later pipeline stages,
     * - transfer reduced-field information back into simulation-owned buffers.
     *
     * The exact semantics are implementation-defined and may range from a no-op
     * placeholder to a full reconstruction or regime-projection step.
     *
     * @param particle_probe Probe exposing the active fluid/particle state.
     * @param domain_probe Probe exposing domain field data and grid metadata.
     * @param searcher_probe Probe exposing spatial neighborhood indexing state.
     * @param codec_probe Mutable codec-side probe carrying codec-specific device state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder that can be used to stage
     * construction parameters before building a codec instance.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return the runtime codec identifier.
     *
     * @details
     * Identifies this concrete implementation as the Knudsen codec family.
     *
     * @return @ref CodecType::knudsen.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;

private:
    /**
     * @brief Reference length scale used during Knudsen-related evaluation.
     *
     * @details
     * Stores the characteristic macroscopic length used to normalize, scale,
     * or interpret local state when computing the codec's Knudsen-derived field.
     *
     * The exact physical meaning depends on the implementation and problem setup.
     */
    T _characteristic_length = T(1);

    /**
     * @brief Device-resident storage for the per-cell Knudsen field.
     *
     * @details
     * Holds the codec-managed scalar field produced by @ref encode and consumed
     * by @ref decode or by other downstream logic.
     *
     * The field is typically aligned with the domain's cell indexing, though the
     * exact layout and sizing policy are implementation-defined.
     */
    DeviceBuffer<T> d_knudsen_values {};
};

/**
 * @brief Fluent builder for @ref KnudsenCodec.
 *
 * @details
 * This builder provides a controlled construction path for @ref KnudsenCodec.
 *
 * It stages the dependencies required to construct the codec, most notably:
 * - the simulation domain,
 * - the characteristic length scale.
 *
 * After validation, the builder can create either:
 * - a codec value via @ref build, or
 * - a host-owned shared pointer via @ref make_host_shared.
 *
 * ## Typical usage
 * @code
 * auto codec = atlas::KnudsenCodecF::builder()
 *     .with_domain(domain)
 *     .with_characteristic_length(0.01f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * The exact validation logic is implementation-defined in `knudsen_codec.hpp`,
 * but it typically checks that:
 * - a required domain has been supplied,
 * - the characteristic length is valid for use,
 * - staged state is sufficiently complete for construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with default-initialized staged state, including no bound
     * domain and a unit characteristic length.
     */
    Builder() = default;

    /**
     * @brief Bind the simulation domain used by the codec instance.
     *
     * @details
     * Stores the supplied host-side domain pointer so it can be forwarded into
     * the final @ref KnudsenCodec during construction.
     *
     * @param domain Host-side shared pointer to the simulation domain.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Set the characteristic length used for Knudsen evaluation.
     *
     * @details
     * Stages the reference physical or numerical length scale used by the codec
     * when evaluating its Knudsen-derived field.
     *
     * @param characteristic_length Reference length scale.
     * @return `*this` for fluent chaining.
     *
     * @note
     * The expected admissible range, such as strictly positive values, is
     * implementation-defined by the validation policy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    /**
     * @brief Build a @ref KnudsenCodec value after validation.
     *
     * @details
     * Validates the staged builder state and constructs a codec object by value.
     *
     * @return Fully constructed codec value.
     *
     * @note
     * Validation behavior is implementation-defined in `knudsen_codec.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    /**
     * @brief Build a host-shared @ref KnudsenCodec after validation.
     *
     * @details
     * Validates the staged builder state, constructs the codec, and returns it
     * inside a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<KnudsenCodec<T>>` owning the constructed codec.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the currently staged builder inputs.
     *
     * Typical checks may include:
     * - verifying that a domain has been provided,
     * - verifying that the characteristic length is acceptable,
     * - rejecting incomplete or inconsistent staged state.
     *
     * @note
     * The exact validation policy is implementation-defined in
     * `knudsen_codec.hpp`.
     */
    void
    validate() const;

private:
    /**
     * @brief Staged domain dependency forwarded into the final codec instance.
     *
     * @details
     * Holds the host-side domain pointer supplied through @ref with_domain so it
     * can be transferred into the constructed @ref KnudsenCodec object.
     */
    DomainHostPtr<T> _domain {};

    /**
     * @brief Staged characteristic length stored until construction time.
     *
     * @details
     * Stores the user-supplied reference length until @ref build or
     * @ref make_host_shared materializes the final codec instance.
     */
    T _characteristic_length = T(1);
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::KnudsenCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using KnudsenCodec = system::KnudsenCodec<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::KnudsenCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<system::KnudsenCodec<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::KnudsenCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<system::KnudsenCodec<T>>;

} // namespace atlas

#include <atlas/codec/knudsen_codec.hpp>