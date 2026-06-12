#pragma once

/**
 * @file deep_learning_codec.h
 * @brief Declares the DeepLearningCodec class, a Codec specialization intended for deep-learning-based encoding and decoding.
 */

#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas {

/**
 * @brief Codec specialization intended for deep-learning-based representations.
 *
 * This class derives from Codec<T> and binds together:
 * - a universe/domain object,
 * - a fluid object,
 * - a spatial hashing searcher.
 *
 * It is intended to serve as an extension point for encoding and decoding
 * simulation data using deep-learning-oriented representations or pipelines.
 *
 * In the current implementation, encode() and decode() are defined but do not
 * yet perform any work.
 *
 * @tparam T Scalar type used by the associated simulation objects.
 */
template <typename T>
class DeepLearningCodec final : public Codec<T> {
public:
    /**
     * @brief Builder for configuring and constructing DeepLearningCodec instances.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    DeepLearningCodec() = default;

    /**
     * @brief Constructs a DeepLearningCodec from its required dependencies.
     *
     * The constructor forwards the provided universe, fluid, and searcher to
     * the base Codec<T> and then resets the codec state.
     *
     * @param domain Host-side shared pointer to the universe/domain object.
     * @param fluid Host-side shared pointer to the fluid object.
     * @param searcher Host-side shared pointer to the spatial hashing searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DeepLearningCodec(UniverseHostPtr<T> domain,
                      FluidHostPtr<T> fluid,
                      SpatialHashingSearcherHostPtr<T> searcher);

    /**
     * @brief Destructor.
     */
    ~DeepLearningCodec() override = default;

    /**
     * @brief Encodes the current simulation state.
     *
     * This override is currently present as an extension point and performs no
     * work in the current implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode() override;

    /**
     * @brief Decodes the current simulation state.
     *
     * This override is currently present as an extension point and performs no
     * work in the current implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode() override;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent DeepLearningCodec construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

private:
};

/**
 * @brief Builder for DeepLearningCodec.
 *
 * This builder collects the required dependencies for constructing a
 * DeepLearningCodec:
 * - universe/domain
 * - fluid
 * - spatial hashing searcher
 *
 * Validation ensures that all required shared pointers are non-null before
 * construction.
 *
 * @tparam T Scalar type used by the associated simulation objects.
 */
template <typename T>
class DeepLearningCodec<T>::Builder final {
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
     * @brief Sets solver indices used for fixed-region cells.
     *
     * The buffer must be empty or match the universe cell count.
     *
     * @param fixed_solver Per-cell solver indices for fixed-region cells.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    /**
     * @brief Sets the fixed-region mask.
     *
     * Cells with value `1` are fixed to the matching solver index from the
     * fixed solver buffer when a derived implementation decodes them.
     *
     * @param fixed_region Per-cell fixed-region mask.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    /**
     * @brief Builds a DeepLearningCodec instance.
     *
     * @return Constructed DeepLearningCodec object.
     *
     * @throw std::invalid_argument Thrown if any required dependency is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeepLearningCodec<T>
    build() const;

    /**
     * @brief Builds a host-side shared DeepLearningCodec instance.
     *
     * @return Host shared pointer to the constructed DeepLearningCodec object.
     *
     * @throw std::invalid_argument Thrown if any required dependency is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DeepLearningCodec<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder state.
     *
     * Ensures that the universe/domain, fluid, and searcher dependencies are all valid.
     *
     * @throw std::invalid_argument Thrown if any required dependency is null.
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
     * @brief Per-cell solver indices for fixed-region cells.
     */
    DeviceBuffer<int> _fixed_solver {};

    /**
     * @brief Per-cell fixed-region mask.
     */
    DeviceBuffer<int> _fixed_region {};
};

} // namespace atlas

namespace atlas {


/**
 * @brief Host-side shared pointer alias for DeepLearningCodec.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using DeepLearningCodecHostPtr = atlas::host_shared_ptr<DeepLearningCodec<T>>;

/**
 * @brief Device-side shared pointer alias for DeepLearningCodec.
 *
 * @tparam T Scalar type used by the codec.
 */
template <typename T>
using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<DeepLearningCodec<T>>;

} // namespace atlas

#include <atlas/codec/deep_learning_codec.hpp>
