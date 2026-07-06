#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/sink/detail/sink_probe_builder.h>

namespace atlas::detail {

bool
SinkProbeBuilder::make(const FluidHostPtr& fluid,
                       const DeviceBuffer<Unit>& units,
                       const DeviceBuffer<atlas::AABB>& unit_bounds,
                       const DeviceBuffer<Despawn>& despawn_operators,
                       const bool flip,
                       const float tolerance,
                       const float dt,
                       SinkProbe& probe) noexcept {
    probe = {};

    if (!fluid || units.empty() || despawn_operators.empty()) {
        return false;
    }

    auto* position_state = fluid->state<atlas::FluidPositionState>();
    auto* active_state   = fluid->state<atlas::FluidActiveState>();

    if (position_state == nullptr || active_state == nullptr) {
        return false;
    }

    auto& positions      = position_state->data();
    auto& active         = active_state->data();
    auto* velocity_state = fluid->state<atlas::FluidVelocityState>();

    if (positions.empty() || active.empty() || fluid->particle_count() == 0 || unit_bounds.empty()) {
        return false;
    }

    probe.units             = atlas::raw_pointer_cast(units.data());
    probe.unit_bounds       = atlas::raw_pointer_cast(unit_bounds.data());
    probe.despawn_operators = atlas::raw_pointer_cast(despawn_operators.data());
    probe.positions         = atlas::raw_pointer_cast(positions.data());

    if (velocity_state != nullptr && !velocity_state->data().empty()) {
        probe.velocities = atlas::raw_pointer_cast(velocity_state->data().data());
    }

    probe.active                 = atlas::raw_pointer_cast(active.data());
    probe.unit_count             = static_cast<int>(units.size());
    probe.despawn_operator_count = static_cast<int>(despawn_operators.size());
    probe.particle_count         = fluid->particle_count();
    probe.flip                   = flip;
    probe.tolerance              = tolerance;
    probe.time_step              = dt;

    return true;
}

}
