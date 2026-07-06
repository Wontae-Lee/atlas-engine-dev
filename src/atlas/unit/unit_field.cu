#include <atlas/unit/unit_field.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/transform/transform_reduce.h>

#include <utility>

namespace atlas {

namespace {

    // transform_reduce operators as functor structs, not `[] ATLAS_ALL_DEVICE`
    // lambdas: an nvcc extended-lambda reduce operator is wrapped in
    // `__nv_dl_wrapper_t`, whose return type thrust's TBB reduce mis-deduces
    // (it tries to build the accumulator from `int`). A plain struct with an
    // explicit `operator()` return type sidesteps that.
    struct SceneBoundTransform {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
        operator()(const AABB& bound) const { return bound.is_valid() ? bound : AABB {}; }
    };
    struct SceneBoundMerge {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
        operator()(AABB lhs, const AABB& rhs) const {
            lhs.merge(rhs);
            return lhs;
        }
    };
    struct BoundIsValid {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator()(const AABB& bound) const { return bound.is_valid(); }
    };
    struct BoolAnd {
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator()(const bool lhs, const bool rhs) const { return lhs && rhs; }
    };

}

UnitField::UnitField(DeviceBuffer<Unit> units)
    : _units(std::move(units)) {
}

void
UnitField::advance(const float dt) {
    if (_units.empty() || !(dt > 0.0f)) {
        return;
    }

    auto* units_ptr = atlas::raw_pointer_cast(_units.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units_ptr, dt] ATLAS_ALL_DEVICE(const int unit_index) {
            units_ptr[unit_index].update(dt);
        });
}

void
UnitField::refresh_bounds() {
    if (_unit_bounds.size() != _units.size()) {
        _unit_bounds.resize(_units.size());
    }

    auto* units_ptr      = atlas::raw_pointer_cast(_units.data());
    auto* bounds_ptr     = atlas::raw_pointer_cast(_unit_bounds.data());
    const int unit_count = static_cast<int>(_units.size());

    // Each unit's geometry bound is in its local frame; Unit::world_bound()
    // re-bounds it in world space through the unit's current sync transform.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        [units_ptr, bounds_ptr] ATLAS_ALL_DEVICE(const int unit_index) {
            bounds_ptr[unit_index] = units_ptr[unit_index].world_bound();
        });

    // Invalid (e.g. unbounded plane) unit bounds are excluded from the
    // scene-level merge so a single unbounded unit does not make
    // scene_bound() useless as a broad-phase reject test.
    _scene_bound = atlas::transform_reduce<ExecutionPolicy::device>(
        _unit_bounds.begin(), _unit_bounds.end(), Bound {},
        SceneBoundTransform {}, SceneBoundMerge {});

    // covers_units() is only true if every unit contributed a valid bound;
    // a single unbounded unit makes the scene-level early-out unsafe.
    _covers_units = atlas::transform_reduce<ExecutionPolicy::device>(
        _unit_bounds.begin(), _unit_bounds.end(), true,
        BoundIsValid {}, BoolAnd {});
}

const DeviceBuffer<Unit>&
UnitField::units() const noexcept {
    return _units;
}

const DeviceBuffer<UnitField::Bound>&
UnitField::unit_bounds() const noexcept {
    return _unit_bounds;
}

const UnitField::Bound&
UnitField::scene_bound() const noexcept {
    return _scene_bound;
}

bool
UnitField::covers_units() const noexcept {
    return _covers_units;
}

std::size_t
UnitField::size() const noexcept {
    return _units.size();
}

bool
UnitField::empty() const noexcept {
    return _units.empty();
}

}
