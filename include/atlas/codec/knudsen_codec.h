#pragma once

#include <atlas/container/container.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>

namespace atlas {

// Reads each cell's number particle count, turns it into a Knudsen number, and
// buckets that number into a solver index — in one pass over every cell.
class KnudsenCodec final {
public:
    class Builder;
    static constexpr int split_count = 4;
    using SplitTable = Container<float, static_cast<std::size_t>(split_count)>;

public:
    KnudsenCodec() = default;

    ATLAS_HOST
    KnudsenCodec(float representative_characteristic_length,
                 float representative_collision_cross_sectional_area,
                 float representative_statistical_weight,
                 float representative_cell_volume);

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    allocate(const UniverseTemperatureState* temperature,
             const UniverseNumberParticleState* number_particle,
             UniverseAllocatedSolverState* allocated_solver) const;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    knudsen_number(const float particle_count) const noexcept {
        const float number_density = _representative_cell_volume > 0.0f
            ? particle_count * _representative_statistical_weight / _representative_cell_volume
            : 0.0f;

        if (!(number_density > 0.0f)
            || !(_representative_characteristic_length > 0.0f)
            || !(_representative_collision_cross_sectional_area > 0.0f)) {
            return 0.0f;
        }

        const float mean_free_path = 1.0f
            / (atlas::SQRT_TWO
               * number_density
               * _representative_collision_cross_sectional_area);
        return mean_free_path / _representative_characteristic_length;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    solver_index(const float kn) const noexcept {
        int index = 0;
        while (index < split_count && !(kn < _kn_split[static_cast<std::size_t>(index)])) {
            ++index;
        }
        return index;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    representative_characteristic_length() const noexcept {
        return _representative_characteristic_length;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    representative_collision_cross_sectional_area() const noexcept {
        return _representative_collision_cross_sectional_area;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    representative_statistical_weight() const noexcept {
        return _representative_statistical_weight;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    representative_cell_volume() const noexcept {
        return _representative_cell_volume;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const SplitTable&
    kn_split() const noexcept {
        return _kn_split;
    }

private:
    float _representative_characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    float _representative_statistical_weight = 1.0f;

    float _representative_cell_volume = 1.0f;

    SplitTable _kn_split { 0.01f, 0.1f, 1.0f, 10.0f };
};

class KnudsenCodec::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_representative_characteristic_length(float representative_characteristic_length) noexcept;

    ATLAS_HOST Builder&
    with_representative_collision_cross_sectional_area(
        float representative_collision_cross_sectional_area) noexcept;

    ATLAS_HOST Builder&
    with_representative_statistical_weight(float representative_statistical_weight) noexcept;

    ATLAS_HOST Builder&
    with_representative_cell_volume(float representative_cell_volume) noexcept;

    ATLAS_NODISCARD ATLAS_HOST KnudsenCodec
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<KnudsenCodec>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    float _representative_characteristic_length = 1.0f;

    float _representative_collision_cross_sectional_area = 1.0f;

    float _representative_statistical_weight = 1.0f;

    float _representative_cell_volume = 1.0f;
};

using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec>;

using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec>;

}
