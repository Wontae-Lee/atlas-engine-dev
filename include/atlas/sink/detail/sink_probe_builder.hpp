#pragma once

namespace atlas::detail {

template <typename T>
bool
SinkProbeBuilder<T>::build(const FluidHostPtr<T>& fluid,
                           const DeviceBuffer<Unit<T>>& units,
                           const DeviceBuffer<atlas::AxisAlignedBoundingBox<T>>& unit_bounds,
                           const DeviceBuffer<DespawnOperator<T>>& despawn_operators,
                           const bool flip,
                           const T tolerance,
                           const T dt,
                           SinkProbe<T>& probe) const noexcept {
    probe = {};

    if (!fluid || units.empty() || despawn_operators.empty()) {
        return false;
    }

    auto* position_state = fluid->template state<atlas::FluidPositionState<T>>();
    auto* active_state   = fluid->template state<atlas::FluidActiveState<T>>();

    if (position_state == nullptr || active_state == nullptr) {
        return false;
    }

    auto& positions      = position_state->data();
    auto& active         = active_state->data();
    auto* velocity_state = fluid->template state<atlas::FluidVelocityState<T>>();

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