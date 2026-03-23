#pragma once

#include <atlas/codec/codec.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

/**
 * @brief Trivial pass-through codec that performs no work.
 *
 * @details
 * `SingleCodec` is a minimal `Codec` implementation intended for cases where a
 * codec object is required by an interface, but no encoding/decoding step is
 * actually desired.
 *
 * What it does:
 * - `encode()` is a no-op (does not modify particle/domain data and does not
 *   write to `codec_probe`).
 * - `decode()` is a no-op (same behavior).
 *
 * Typical uses:
 * - Baseline for profiling (measuring overhead without codec work).
 * - Placeholder/default codec in pipelines where the “codec stage” is optional.
 * - Testing and integration when only the system wiring is needed.
 *
 * Important notes:
 * - The codec still inherits the base `Codec<T>` behavior. In particular, the
 *   base class may allocate/initialize internal buffers tied to the domain.
 * - A valid domain is required for the domain-dependent sizing performed in
 *   `Codec<T>` (even though this derived class does no work in encode/decode).
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @see Codec
 * @see CodecDeviceProbe
 */
template <typename T>
class SingleCodec final : public Codec<T> {
public:
    /**
     * @brief Fluent builder for constructing a `SingleCodec`.
     *
     * @details
     * Use `builder()` to obtain a builder, then configure required fields such as
     * the domain, and call `build()` or `make_host_shared()`.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an instance without binding a domain. This is mainly useful for
     * containers/serialization or when the object is immediately replaced.
     *
     * @warning
     * The codec is not ready for use until it is constructed with a valid domain
     * (because `Codec<T>` sizing depends on the domain).
     */
    SingleCodec() = default;

    /**
     * @brief Construct a no-op codec bound to a domain.
     *
     * @details
     * Forwards the domain to the base `Codec<T>` constructor, which validates
     * the pointer and initializes domain-sized storage used by the codec system.
     *
     * @param domain Host pointer to a domain (must be non-null).
     *
     * @pre `domain != nullptr`
     *
     * @throws std::invalid_argument If `domain` is null (thrown by the base class).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SingleCodec(const DomainHostPtr<T>& domain);

    /// @brief Virtual destructor.
    ~SingleCodec() override = default;

    /**
     * @brief Encode step (no-op).
     *
     * @details
     * This implementation intentionally performs no encoding work. It does not:
     * - read or modify `particle_probe` data,
     * - read or modify `domain_probe` data,
     * - depend on `searcher_probe`,
     * - write to `codec_probe`.
     *
     * The function exists to satisfy the `Codec<T>` interface and to allow
     * plugging this codec into pipelines that expect an encode stage.
     *
     * @param particle_probe Particle data probe (unused).
     * @param domain_probe   Domain data probe (unused).
     * @param searcher_probe Spatial hash probe (unused).
     * @param codec_probe    Codec device probe (unused).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Decode step (no-op).
     *
     * @details
     * This implementation intentionally performs no decoding work. It does not:
     * - read or modify `particle_probe` data,
     * - read or modify `domain_probe` data,
     * - depend on `searcher_probe`,
     * - write to `codec_probe`.
     *
     * The function exists to satisfy the `Codec<T>` interface and to allow
     * plugging this codec into pipelines that expect a decode stage.
     *
     * @param particle_probe Particle data probe (unused).
     * @param domain_probe   Domain data probe (unused).
     * @param searcher_probe Spatial hash probe (unused).
     * @param codec_probe    Codec device probe (unused).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode(const ParticleDeviceProbe<T>& particle_probe,
           const DomainDeviceProbe<T>& domain_probe,
           const SpatialHashingProbe<T>& searcher_probe,
           CodecDeviceProbe<T>& codec_probe) override;

    /**
     * @brief Create a builder for `SingleCodec`.
     *
     * @return A default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return the runtime codec tag for this no-op implementation.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CodecType
    type() const noexcept override;
};

/**
 * @brief Builder for `SingleCodec<T>`.
 *
 * @details
 * Provides a fluent interface to configure and construct a `SingleCodec`.
 *
 * Required configuration:
 * - `with_domain(...)` must be called with a non-null domain pointer.
 *
 * Construction forms:
 * - `build()` returns the codec by value.
 * - `make_host_shared()` returns `host_shared_ptr<SingleCodec<T>>`.
 *
 * Validation:
 * - `validate()` enforces required configuration and is called by
 *   `build()` and `make_host_shared()`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class SingleCodec<T>::Builder final {
public:
    /// @brief Construct an empty builder.
    Builder() = default;

    /**
     * @brief Bind the domain used by the base `Codec<T>`.
     *
     * @param domain Host pointer to a domain (must be non-null).
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Construct the codec (returned by value).
     *
     * @return A configured `SingleCodec<T>`.
     *
     * @throws std::invalid_argument If required configuration is missing or invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SingleCodec<T>
    build() const;

    /**
     * @brief Construct the codec as a host-shared pointer.
     *
     * @return `host_shared_ptr<SingleCodec<T>>` owning the constructed codec.
     *
     * @throws std::invalid_argument If required configuration is missing or invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SingleCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate builder state.
     *
     * @throws std::invalid_argument If `_domain` is null.
     */
    void
    validate() const;

private:
    /// Domain required by the base `Codec<T>` sizing/initialization.
    DomainHostPtr<T> _domain {};
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using SingleCodec = system::SingleCodec<T>;

template <typename T>
using SingleCodecHostPtr = atlas::host_shared_ptr<system::SingleCodec<T>>;

template <typename T>
using SingleCodecDevicePtr = atlas::device_shared_ptr<system::SingleCodec<T>>;

} // namespace atlas

// Template implementation.
#include <atlas/codec/single_codec.hpp>
