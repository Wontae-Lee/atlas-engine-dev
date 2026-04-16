#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcSolver<T>::DsmcSolver(atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
                          DsmcOperator<T> op)
    : _fluid(std::move(fluid))
    , _operator(std::move(op)) {
    rebuild_pair_tables();
}

template <typename T>
void
DsmcSolver<T>::set_operator(DsmcOperator<T> op) noexcept {
    _operator = std::move(op);
}

template <typename T>
void
DsmcSolver<T>::set_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) {
    _fluid = std::move(fluid);
    rebuild_pair_tables();
}

template <typename T>
const atlas::host_shared_ptr<atlas::Fluid<T>>&
DsmcSolver<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
const DsmcOperator<T>&
DsmcSolver<T>::collision_operator() const noexcept {
    return _operator;
}

template <typename T>
DsmcOperator<T>&
DsmcSolver<T>::collision_operator() noexcept {
    return _operator;
}

template <typename T>
void
DsmcSolver<T>::apply_field_force(Universe<T>& domain,
                                 FluidDeviceProbe<T> particle) const {
    if (domain.field_force == nullptr || particle.pos == nullptr || particle.vel == nullptr || particle.particle_count <= 0) {
        return;
    }

    auto* positions               = particle.pos;
    auto* velocities              = particle.vel;
    const auto* field_force       = domain.field_force;
    const Vector3<T> lower_corner = domain.lower_corner;
    const Vector3<int> grid_size  = domain.grid_size;
    const T inv_h                 = domain.inv_h;
    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle.particle_count,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<T> rel = (positions[i] - lower_corner) * inv_h;
            Vector3<int> ijk     = atlas::math::floor(rel).template cast_to<int>();
            ijk                  = atlas::math::clamp(ijk, lo, hi);

            const int cell = ijk.x + ijk.y * grid_size.x + ijk.z * grid_size.x * grid_size.y;
            velocities[i] += field_force[cell];
        });
}

template <typename T>
void
DsmcSolver<T>::rebuild_pair_tables() {
    if (!_fluid) {
        throw std::runtime_error("DsmcSolver: fluid must not be null.");
    }

    _species_count = _fluid->size();
    if (_species_count <= 0) {
        _effective_collision_diameters.clear();
        _effective_viscosity_indices.clear();
        _effective_scattering_parameters.clear();
        _reduced_masses.clear();
        return;
    }

    const std::size_t pair_count = static_cast<std::size_t>(_species_count) * static_cast<std::size_t>(_species_count + 1) / 2;

    HostBuffer<MatrialProperties<T>> particles_host(static_cast<std::size_t>(_species_count));
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(_fluid->particles().data()),
        particles_host.data(),
        particles_host.size());

    HostBuffer<T> effective_collision_diameters_host(pair_count, T(0));
    HostBuffer<T> effective_viscosity_indices_host(pair_count, T(1));
    HostBuffer<T> effective_scattering_parameters_host(pair_count, T(1));
    HostBuffer<T> reduced_masses_host(pair_count, T(0));

    for (int s = 0; s < _species_count; ++s) {
        for (int r = s; r < _species_count; ++r) {
            const int pair_index = _pair_indexer.pair_index(s, r, _species_count);
            const auto& a        = particles_host[static_cast<std::size_t>(s)];
            const auto& b        = particles_host[static_cast<std::size_t>(r)];

            const T da    = a.collision_diameter.has_value() ? *a.collision_diameter : T(0);
            const T db    = b.collision_diameter.has_value() ? *b.collision_diameter : T(0);
            const T wa    = a.viscosity_index.has_value() ? *a.viscosity_index : T(1);
            const T wb    = b.viscosity_index.has_value() ? *b.viscosity_index : T(1);
            const T aa    = a.scattering_parameter.has_value() ? *a.scattering_parameter : T(1);
            const T ab    = b.scattering_parameter.has_value() ? *b.scattering_parameter : T(1);
            const T denom = a.mass + b.mass;

            effective_collision_diameters_host[static_cast<std::size_t>(pair_index)]   = T(0.5) * (da + db);
            effective_viscosity_indices_host[static_cast<std::size_t>(pair_index)]     = T(0.5) * (wa + wb);
            effective_scattering_parameters_host[static_cast<std::size_t>(pair_index)] = T(0.5) * (aa + ab);
            reduced_masses_host[static_cast<std::size_t>(pair_index)]                  = (a.mass > T(0) && b.mass > T(0) && denom > T(0)) ? ((a.mass * b.mass) / denom) : T(0);
        }
    }

    _effective_collision_diameters = DeviceBuffer<T>(
        effective_collision_diameters_host.begin(),
        effective_collision_diameters_host.end());
    _effective_viscosity_indices = DeviceBuffer<T>(
        effective_viscosity_indices_host.begin(),
        effective_viscosity_indices_host.end());
    _effective_scattering_parameters = DeviceBuffer<T>(
        effective_scattering_parameters_host.begin(),
        effective_scattering_parameters_host.end());
    _reduced_masses = DeviceBuffer<T>(
        reduced_masses_host.begin(),
        reduced_masses_host.end());
}

}