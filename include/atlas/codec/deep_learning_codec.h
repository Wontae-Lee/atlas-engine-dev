#pragma once

#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas {

template <typename T>
class DeepLearningCodec final : public Codec<T> {
public:
    class Builder;

public:
    DeepLearningCodec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DeepLearningCodec(UniverseHostPtr<T> domain,
                      FluidHostPtr<T> fluid,
                      SearcherHostPtr<T> searcher);

    ~DeepLearningCodec() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode() override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode() override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

private:
};

template <typename T>
class DeepLearningCodec<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(UniverseHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeepLearningCodec<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DeepLearningCodec<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    UniverseHostPtr<T> _domain {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    DeviceBuffer<int> _fixed_solver {};

    DeviceBuffer<int> _fixed_region {};
};

}

namespace atlas {

template <typename T>
using DeepLearningCodecHostPtr = atlas::host_shared_ptr<DeepLearningCodec<T>>;

template <typename T>
using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<DeepLearningCodec<T>>;

}

#include <atlas/codec/deep_learning_codec.hpp>