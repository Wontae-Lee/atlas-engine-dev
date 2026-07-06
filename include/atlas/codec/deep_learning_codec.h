#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

/**
 * @file deep_learning_codec.h
 * @brief Placeholder `Codec` for a future learned (rather than
 *        analytic-formula) per-cell solver classifier.
 *
 * @details
 * Where `KnudsenCodec` decides each cell's solver from a closed-form
 * physical criterion (Knudsen number), a learned classifier could
 * instead predict the appropriate solver from a trained model over
 * richer per-cell features — potentially capturing regime transitions a
 * single scalar threshold can't. This class currently only establishes
 * the `Codec` interface shape (constructor, `Builder`, the
 * `fixed_solver`/`fixed_region` override plumbing inherited from
 * `Codec`); `encode()`/`decode()` are stubs that just rebuild the probe
 * (`make_probe()`) without computing or writing any classification —
 * `allocated_solver()` is left at whatever `Codec::reset()` initialized
 * it to (all-zero) until an actual model integration lands.
 */

namespace atlas {

/**
 * @brief Scaffold for a learned per-cell solver classifier; `encode()`/
 *        `decode()` do not yet compute a classification. See this
 *        file's top-of-file documentation.
 */
class DeepLearningCodec final : public Codec {
public:
    class Builder;

    DeepLearningCodec() = default;

    ATLAS_HOST
    DeepLearningCodec(UniverseHostPtr domain,
                      FluidHostPtr fluid,
                      SearcherHostPtr searcher);

    ~DeepLearningCodec() override = default;

    /** @brief Currently a stub: rebuilds the probe only, does not write
     *  any classification (see this file's top-of-file documentation). */
    ATLAS_HOST void
    encode() override;

    /** @brief Currently a stub: rebuilds the probe only, does not write
     *  any classification (see this file's top-of-file documentation). */
    ATLAS_HOST void
    decode() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;
};

/**
 * @brief Fluent builder for `DeepLearningCodec`. Validation requires
 *        non-null `_domain`/`_fluid`/`_searcher` and
 *        `_fixed_solver`/`_fixed_region` either empty or sized to the
 *        domain's cell count.
 */
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
