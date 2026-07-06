#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

namespace atlas {

class KnudsenCodec final : public Codec {
public:
    class Builder;

    KnudsenCodec() = default;

    ATLAS_HOST
    KnudsenCodec(UniverseHostPtr domain,
                 FluidHostPtr fluid,
                 SearcherHostPtr searcher,
                 float characteristic_length,
                 float representative_collision_cross_sectional_area = 1.0f);

    ~KnudsenCodec() override = default;

    ATLAS_HOST void
    encode() override;

    ATLAS_HOST void
    decode() override;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

public:
    ATLAS_HOST void
    encode_cells();

    ATLAS_HOST void
    decode_cells();

private:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    fixed_cell(const CodecProbe& probe, const int cell) noexcept {
        return probe.fixed_region_ptr != nullptr && probe.fixed_region_ptr[cell] == 1;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    knudsen_number(const float particle_count,
                   const float statistical_weight,
                   const float cell_volume,
                   const float characteristic_length,
                   const float representative_collision_cross_sectional_area) noexcept {
        const float number_density = cell_volume > 0.0f
            ? particle_count * statistical_weight / cell_volume
            : 0.0f;

        if (!(number_density > 0.0f)
            || !(characteristic_length > 0.0f)
            || !(representative_collision_cross_sectional_area > 0.0f)) {
            return 0.0f;
        }

        const float mean_free_path = 1.0f
            / (atlas::SQRT_TWO
               * number_density
               * representative_collision_cross_sectional_area);
        return mean_free_path / characteristic_length;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static int
    solver_index(const float kn, const float* splits, const int split_count) noexcept {
        int index = 0;
        while (index < split_count && !(kn < splits[index])) {
            ++index;
        }
        return index;
    }

private:
    float _characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    DeviceBuffer<float> d_kn_split {};
};

class KnudsenCodec::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_domain(UniverseHostPtr domain) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_characteristic_length(float characteristic_length) noexcept;

    ATLAS_HOST Builder&
    with_representative_collision_cross_sectional_area(
        float representative_collision_cross_sectional_area) noexcept;

    ATLAS_HOST Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    ATLAS_HOST Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    ATLAS_NODISCARD ATLAS_HOST KnudsenCodec
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<KnudsenCodec>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _domain {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    float _characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    DeviceBuffer<int> _fixed_solver {};

    DeviceBuffer<int> _fixed_region {};
};

using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec>;

using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec>;

}
