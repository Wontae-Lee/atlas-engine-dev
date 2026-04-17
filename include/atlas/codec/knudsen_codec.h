#pragma once

/**
 * @file knudsen_codec.h
 * @brief Declares the KnudsenCodec class for Knudsen-number-related encoding over a simulation domain.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Codec specialization for Knudsen-number-related field encoding.
 *
 * This class derives from Codec<T> and binds together:
 * - a universe/domain object,
 * - a fluid object,
 * - a spatial hashing searcher,
 * - a characteristic length scale used in Knudsen-number interpretation.
 *
 * The class owns a device buffer of per-cell Knudsen-related values, intended
 * to store domain-discretized results associated with the codec.
 *
 * In the current implementation, encode() and decode() are declared and defined
 * but do not yet perform any computation.
 *
 * @tparam T Scalar type used by the associated simulation objects.
 */
template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    /**
     * @brief Builder for configuring and constructing KnudsenCodec instances.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    KnudsenCodec() = default;

    /**
     * @brief Constructs a KnudsenCodec from its required dependencies.
     *
     * The constructor forwards the provided universe, fluid, and searcher to the
     * base Codec<T>, stores the characteristic length, validates it, and
     * allocates per-cell device storage for Knudsen-related values.
     *
     * @param domain Host-side shared pointer to the universe/domain object.
     * @param fluid Host-side shared pointer to the fluid object.
     * @param searcher Host-side shared pointer to the spatial hashing searcher.
     * @param characteristic_length Positive characteristic length scale.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    KnudsenCodec(UniverseHostPtr<T> domain,
                 FluidHostPtr<T> fluid,
                 SpatialHashingSearcherHostPtr<T> searcher,
                 T characteristic_length);

    /**
     * @brief Destructor.
     */
    ~KnudsenCodec() override = default;

    /**
     * @brief Encodes Knudsen-related information into the codec buffer.
     *
     * This override is currently present as an extension point and performs no
     * work in the current implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode() override;

    /**
     * @brief Decodes Knudsen-related information from the codec buffer.
     *
     * This override is currently present as an extension point and performs no
     * work in the current implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode() override;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent KnudsenCodec construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

private:
    /**
     * @brief Characteristic length scale used by the codec.
     *
     * This value is expected to be strictly positive and is typically used in
     * Knudsen-number-related normalization or interpretation.
     */
    T _characteristic_length = T(1);

    /**
     * @brief Device buffer storing per-cell Knudsen-related values.
     *
     * The buffer is sized to the number of cells in the associated universe.
     */
    DeviceBuffer<T> d_knudsen_values {};
};

/**
 * @brief Builder for KnudsenCodec.
 *
 * This builder collects the dependencies and configuration required to create a
 * KnudsenCodec:
 * - universe/domain
 * - fluid
 * - spatial hashing searcher
 * - characteristic length
 *
 * Validation ensures that required shared pointers are non-null and that the
 * characteristic length is strictly positive.
 *
 * @tparam T Scalar type used by the associated simulation objects.
 */
template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the universe/domain dependency.
     *
     * @param domain Host-side shared pointer to the universe/domain object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(UniverseHostPtr<T> domain) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @param fluid Host-side shared pointer to the fluid object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @param searcher Host-side shared pointer to the searcher object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the characteristic length scale.
     *
     * The value must be strictly positive.
     *
     * @param characteristic_length Characteristic length scale.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    /**
     * @brief Builds a KnudsenCodec instance.
     *
     * @return Constructed KnudsenCodec object.
     *
     * @throw std::invalid_argument Thrown if any required dependency is null or
     *         if the characteristic length is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    /**
     * @brief Builds a host-side shared KnudsenCodec instance.
     *
     * @return Host shared pointer to the constructed KnudsenCodec object.
     *
     * @throw std::invalid_argument Thrown if any required dependency is null or
     *         if the characteristic length is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder state.
     *
     * Ensures that the universe/domain, fluid, and searcher dependencies are all
     * valid and that the characteristic length is positive.
     *
     * @throw std::invalid_argument Thrown if validation fails.
     */
    void
    validate() const;

private:
    /**
     * @brief Universe/domain dependency used by the codec.
     */
    UniverseHostPtr<T> _domain {};

    /**
     * @brief Fluid dependency used by the codec.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency used by the codec.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Characteristic length scale used to construct the codec.
     */
    T _characteristic_length = T(1);
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::KnudsenCodec.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using KnudsenCodec = system::KnudsenCodec<T>;

/**
 * @brief Host-side shared pointer alias for KnudsenCodec.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<system::KnudsenCodec<T>>;

/**
 * @brief Device-side shared pointer alias for KnudsenCodec.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<system::KnudsenCodec<T>>;

} // namespace atlas

#include <atlas/codec/knudsen_codec.hpp>