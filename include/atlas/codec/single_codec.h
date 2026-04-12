#pragma once

/**
 * @file single_codec.h
 * @brief Declares the trivial single-representation codec implementation.
 *
 * @details
 * This header defines @ref atlas::system::SingleCodec, the simplest concrete
 * implementation of the @ref atlas::system::Codec interface.
 *
 * Unlike codecs that transform simulation state into an auxiliary, reduced, or
 * model-specific representation, @ref SingleCodec is intended for workflows in
 * which the simulation remains in a single directly resolved representation.
 *
 * In practice, this codec is useful when:
 * - a uniform codec abstraction is required by higher-level orchestration code,
 * - no compressed, latent, or regime-specific representation is needed,
 * - encode/decode stages must still exist as formal pipeline hooks,
 * - experiments or applications want the simplest possible codec behavior.
 *
 * ## Conceptual role
 * `SingleCodec` acts as the identity-like member of the codec family:
 * - @ref encode typically preserves the direct simulation representation,
 * - @ref decode typically restores or forwards that same representation with
 *   little or no transformation,
 * - @ref type identifies the codec as @ref CodecType::single.
 *
 * ## Why this codec exists
 * Even when no representation change is required, keeping a concrete "single"
 * codec provides important architectural benefits:
 * - the simulation pipeline can always depend on a codec object,
 * - runtime codec selection remains uniform,
 * - downstream code does not need to special-case "no codec",
 * - future extensions can preserve the same orchestration contract.
 *
 * ## Construction
 * The codec may be:
 * - default-constructed, or
 * - created through the nested @ref Builder, which stages the required domain
 *   dependency before validation and construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout simulation and codec operations.
 */

#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Minimal codec that keeps the simulation in a single resolved state.
 *
 * @details
 * @ref SingleCodec is the most lightweight concrete specialization of
 * @ref Codec.
 *
 * It is designed for scenarios where simulation data does not need to be mapped
 * into a secondary representation such as:
 * - a reduced-order state,
 * - a compressed latent code,
 * - a diagnostic field,
 * - a learned surrogate representation.
 *
 * Instead, this codec preserves the direct one-to-one simulation state while
 * still providing the formal @ref encode and @ref decode entry points expected
 * by the generic codec pipeline.
 *
 * ## Intended behavior
 * In a typical implementation:
 * - @ref encode acts as an identity-like or bookkeeping step,
 * - @ref decode acts as the symmetric return path required by the interface,
 * - little or no internal codec-specific state is needed beyond the base class.
 *
 * ## Domain association
 * The codec may still be associated with a @ref Domain because the broader codec
 * abstraction expects access to simulation context such as:
 * - grid topology,
 * - persistent field storage,
 * - metadata needed by higher-level systems.
 *
 * ## Runtime type
 * This class reports @ref CodecType::single from @ref type().
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class SingleCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for @ref SingleCodec.
     *
     * @details
     * The builder provides a controlled construction path for the codec by
     * staging required inputs, validating them, and then materializing either:
     * - a value instance of @ref SingleCodec, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a codec with default-initialized state.
     *
     * @note
     * A default-constructed codec may still require a valid domain association
     * before it is meaningful in a larger simulation pipeline.
     */
    SingleCodec() = default;

    /**
     * @brief Construct the codec for a specific simulation domain.
     *
     * @details
     * Associates the codec with the supplied domain so it can participate
     * consistently in the broader simulation and codec orchestration flow.
     *
     * Although this codec does not introduce a richer reduced representation,
     * the domain may still be relevant for:
     * - contextual consistency with other codec implementations,
     * - access to grid-aligned data structures,
     * - integration with systems that expect every codec to know its domain.
     *
     * @param domain Host-side shared pointer to the associated simulation domain.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SingleCodec(const DomainHostPtr<T>& domain);

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Defaulted override for correct destruction through base-class pointers.
     */
    ~SingleCodec() override = default;

    /**
     * @brief Encode the current simulation state without changing representation.
     *
     * @details
     * This function is the forward transformation hook required by the codec
     * interface.
     *
     * For @ref SingleCodec, the encode step conceptually preserves the direct
     * simulation representation rather than projecting it into a reduced,
     * compressed, or alternate form.
     *
     * Depending on the implementation, this function may:
     * - perform no transformation at all,
     * - synchronize or validate state,
     * - act as a formal pipeline checkpoint,
     * - prepare bookkeeping needed for a later decode step.
     *
     * @param particle_probe Probe exposing the active fluid/particle state.
     * @param domain_probe Probe exposing domain field data and grid metadata.
     * @param searcher_probe Probe exposing spatial neighborhood indexing state.
     * @param codec_probe Mutable codec-side probe carrying codec-specific device state.
     *
     * @note
     * The exact behavior is implementation-defined in `single_codec.hpp`, but
     * it is expected to remain minimal compared with other codec types.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const FluidDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decode codec-managed state back into the direct simulation representation.
     *
     * @details
     * This function is the inverse-facing hook required by the generic codec
     * interface.
     *
     * In the single-representation case, decode is typically symmetric with
     * @ref encode and may involve little or no actual transformation.
     *
     * Depending on the implementation, it may:
     * - behave as a no-op,
     * - restore or finalize directly represented state,
     * - satisfy pipeline expectations that all codecs support a decode phase,
     * - provide a future extension point without changing the interface.
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
     * Identifies this concrete implementation as the single-representation codec.
     *
     * @return @ref CodecType::single.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;
};

/**
 * @brief Fluent builder for @ref SingleCodec.
 *
 * @details
 * This builder provides a controlled construction path for @ref SingleCodec.
 *
 * It stages the dependencies required to construct the codec, validates them,
 * and then materializes either:
 * - a codec value via @ref build, or
 * - a host-owned shared pointer via @ref make_host_shared.
 *
 * ## Typical usage
 * @code
 * auto codec = atlas::SingleCodecF::builder()
 *     .with_domain(domain)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * The exact validation logic is implementation-defined in `single_codec.hpp`,
 * but it typically checks that the staged domain dependency is present and
 * acceptable for construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class SingleCodec<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with default-initialized staged state and no bound domain.
     */
    Builder() = default;

    /**
     * @brief Bind the simulation domain used by the codec instance.
     *
     * @details
     * Stores the supplied host-side domain pointer so it can be forwarded into
     * the final @ref SingleCodec during construction.
     *
     * @param domain Host-side shared pointer to the simulation domain.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Build a @ref SingleCodec value after validation.
     *
     * @details
     * Validates the staged builder state and constructs a codec object by value.
     *
     * @return Fully constructed codec value.
     *
     * @note
     * Validation behavior is implementation-defined in `single_codec.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SingleCodec<T>
    build() const;

    /**
     * @brief Build a host-shared @ref SingleCodec after validation.
     *
     * @details
     * Validates the staged builder state, constructs the codec, and returns it
     * inside a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<SingleCodec<T>>` owning the constructed codec.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SingleCodec<T>>
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
     * - rejecting incomplete staged state,
     * - enforcing any construction preconditions defined by the implementation.
     *
     * @note
     * The exact validation policy is implementation-defined in `single_codec.hpp`.
     */
    void
    validate() const;

private:
    /**
     * @brief Staged domain dependency forwarded into the final codec instance.
     *
     * @details
     * Holds the host-side domain pointer supplied through @ref with_domain so it
     * can be transferred into the constructed @ref SingleCodec object.
     */
    DomainHostPtr<T> _domain {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::SingleCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SingleCodec = system::SingleCodec<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::SingleCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SingleCodecHostPtr = atlas::host_shared_ptr<system::SingleCodec<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::SingleCodec.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SingleCodecDevicePtr = atlas::device_shared_ptr<system::SingleCodec<T>>;

} // namespace atlas

#include <atlas/codec/single_codec.hpp>