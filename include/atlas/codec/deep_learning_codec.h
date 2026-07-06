#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

namespace atlas {

class DeepLearningCodec final : public Codec {
public:
    class Builder;

    DeepLearningCodec() = default;

    ATLAS_HOST
    DeepLearningCodec(UniverseHostPtr domain,
                      FluidHostPtr fluid,
                      SearcherHostPtr searcher);

    ~DeepLearningCodec() override = default;

    ATLAS_HOST void
    encode() override;

    ATLAS_HOST void
    decode() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;
};

class DeepLearningCodec::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_domain(UniverseHostPtr domain) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    ATLAS_HOST Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    ATLAS_NODISCARD ATLAS_HOST DeepLearningCodec
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<DeepLearningCodec>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _domain {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    DeviceBuffer<int> _fixed_solver {};

    DeviceBuffer<int> _fixed_region {};
};

using DeepLearningCodecHostPtr = atlas::host_shared_ptr<DeepLearningCodec>;

using DeepLearningCodecDevicePtr = atlas::device_shared_ptr<DeepLearningCodec>;

}
