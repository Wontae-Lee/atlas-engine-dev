#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
SphSolver<T>::SphSolver(FluidHostPtr<T> fluid,
            SphKernel<T> op)
    : _fluid(std::move(fluid))
    , _operator(std::move(op)) {
    rebuild_pair_tables();
}

template <typename T>
typename SphSolver<T>::Builder
SphSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
SphSolver<T>::solve(DomainDeviceProbe<T>& domain,
              SpatialHashingProbe<T>& searcher,
              FluidDeviceProbe<T>& particle,
              CodecDeviceProbe<T>&) {
    if (particle.particle_count <= 0 || particle.pos == nullptr || particle.vel == nullptr || particle.species == nullptr) {
        return;
    }

    const auto interaction_operator = _operator;
    auto* positions                 = particle.pos;
    auto* velocities                = particle.vel;
    const auto* temperatures        = particle.temperature;
    const auto* species             = particle.species;
    const auto* cell_start          = searcher.cell_start;
    const auto* cell_end            = searcher.cell_end;
    const auto* indices             = searcher.indices;
    const auto* field_force         = domain.field_force;
    const auto* rest_densities      = atlas::raw_pointer_cast(_rest_densities.data());
    const auto* pressure_coeffs     = atlas::raw_pointer_cast(_pressure_coefficients.data());
    const auto* dynamic_viscosities = atlas::raw_pointer_cast(_dynamic_viscosities.data());
    const auto pair_indexer         = _pair_indexer;
    const int species_count         = _species_count;
    const Vector3<T> lower_corner   = domain.lower_corner;
    const Vector3<int> grid_size    = domain.grid_size;
    const T inv_h                   = domain.inv_h;
    const T smoothing_length        = searcher.cell_size;

    if (cell_start == nullptr || cell_end == nullptr || indices == nullptr || species_count <= 0 || !(smoothing_length > T(0))) {
        return;
    }

    if (field_force != nullptr) {
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

    const atlas::device_ptr<Vector3<T>> velocity_snapshot_begin(velocities);
    const DeviceBuffer<Vector3<T>> velocity_snapshot(
        velocity_snapshot_begin,
        velocity_snapshot_begin + particle.particle_count);
    const auto* velocity_snapshot_ptr = atlas::raw_pointer_cast(velocity_snapshot.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle.particle_count,
        [=] ATLAS_DEVICE(const int p) {
            const size_t species_p = species[p];
            const T mass_p         = particle.particle_property[species_p].mass;
            const T temperature_p  = (temperatures != nullptr && temperatures[p] > T(0)) ? temperatures[p] : T(1);

            if (!(mass_p > T(0))) {
                return;
            }

            Vector3<T> velocity_delta(T(0), T(0), T(0));

            searcher.for_each_neighbor(
                p,
                positions,
                [=, &velocity_delta] ATLAS_DEVICE(const int q) mutable {
                    if (q < 0 || q >= particle.particle_count) {
                        return false;
                    }

                    const size_t species_q = species[q];
                    const int material_pair_index = pair_indexer.pair_index(
                        static_cast<int>(species_p),
                        static_cast<int>(species_q),
                        species_count);

                    const Vector3<T> dx = positions[q] - positions[p];
                    const T distance    = dx.length();
                    if (!(distance > T(0)) || distance >= smoothing_length) {
                        return false;
                    }

                    const T mass_q = particle.particle_property[species_q].mass;
                    const T rest_density = rest_densities[material_pair_index];
                    if (!(mass_q > T(0)) || !(rest_density > T(0))) {
                        return false;
                    }

                    const T weight            = interaction_operator.weight(distance, smoothing_length);
                    const T grad              = interaction_operator.gradient_factor(distance, smoothing_length);
                    const T lap               = interaction_operator.laplacian(distance, smoothing_length);
                    const T pressure_coeff    = pressure_coeffs[material_pair_index];
                    const T dynamic_viscosity = dynamic_viscosities[material_pair_index];
                    const T temperature_q = (temperatures != nullptr && temperatures[q] > T(0)) ? temperatures[q] : T(1);
                    const T pair_temperature = T(0.5) * (temperature_p + temperature_q);
                    const T pair_density     = (mass_p + mass_q) * weight;
                    const T pressure         = pressure_coeff * pair_temperature * (pair_density - rest_density);
                    const Vector3<T> dir     = dx / distance;
                    const Vector3<T> pressure_force = dir * (-pressure * grad);
                    const Vector3<T> viscosity_force = (velocity_snapshot_ptr[q] - velocity_snapshot_ptr[p])
                        * (dynamic_viscosity * pair_temperature * lap);

                    velocity_delta += (pressure_force + viscosity_force) / mass_p;
                    return false;
                });

            velocities[p] += velocity_delta;
        });
}

template <typename T>
void
SphSolver<T>::set_operator(SphKernel<T> op) noexcept {
    _operator = std::move(op);
}

template <typename T>
void
SphSolver<T>::set_fluid(FluidHostPtr<T> fluid) {
    _fluid = std::move(fluid);
    rebuild_pair_tables();
}

template <typename T>
const FluidHostPtr<T>&
SphSolver<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
const SphKernel<T>&
SphSolver<T>::interaction_operator() const noexcept {
    return _operator;
}

template <typename T>
SphKernel<T>&
SphSolver<T>::interaction_operator() noexcept {
    return _operator;
}

template <typename T>
void
SphSolver<T>::rebuild_pair_tables() {
    if (!_fluid) {
        throw std::runtime_error("SphSolver: fluid must not be null.");
    }

    _species_count = _fluid->size();
    if (_species_count <= 0) {
        _rest_densities.clear();
        _pressure_coefficients.clear();
        _dynamic_viscosities.clear();
        return;
    }

    const std::size_t pair_count =
        static_cast<std::size_t>(_species_count) * static_cast<std::size_t>(_species_count + 1) / 2;

    HostBuffer<MatrialProperties<T>> particles_host(static_cast<std::size_t>(_species_count));
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(_fluid->particles().data()),
        particles_host.data(),
        particles_host.size());

    HostBuffer<T> rest_density_host(pair_count, T(1));
    HostBuffer<T> pressure_coeff_host(pair_count, T(1));
    HostBuffer<T> dynamic_viscosity_host(pair_count, T(0));

    for (int s = 0; s < _species_count; ++s) {
        for (int r = s; r < _species_count; ++r) {
            const int pair_index = _pair_indexer.pair_index(s, r, _species_count);
            const auto& a = particles_host[static_cast<std::size_t>(s)];
            const auto& b = particles_host[static_cast<std::size_t>(r)];

            const T rho_a = a.rest_density.has_value() ? *a.rest_density : T(1);
            const T rho_b = b.rest_density.has_value() ? *b.rest_density : T(1);
            const T k_a   = a.pressure_coefficient.has_value() ? *a.pressure_coefficient : T(1);
            const T k_b   = b.pressure_coefficient.has_value() ? *b.pressure_coefficient : T(1);
            const T mu_a  = a.dynamic_viscosity.has_value() ? *a.dynamic_viscosity : T(0);
            const T mu_b  = b.dynamic_viscosity.has_value() ? *b.dynamic_viscosity : T(0);

            rest_density_host[static_cast<std::size_t>(pair_index)]       = T(0.5) * (rho_a + rho_b);
            pressure_coeff_host[static_cast<std::size_t>(pair_index)]     = T(0.5) * (k_a + k_b);
            dynamic_viscosity_host[static_cast<std::size_t>(pair_index)]  = T(0.5) * (mu_a + mu_b);
        }
    }

    _rest_densities = DeviceBuffer<T>(rest_density_host.begin(), rest_density_host.end());
    _pressure_coefficients = DeviceBuffer<T>(pressure_coeff_host.begin(), pressure_coeff_host.end());
    _dynamic_viscosities = DeviceBuffer<T>(dynamic_viscosity_host.begin(), dynamic_viscosity_host.end());
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_operator(const SphKernel<T>& op) noexcept {
    _operator = op;
    return *this;
}

template <typename T>
void
SphSolver<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("SphSolver::Builder: fluid must not be null.");
    }
}

template <typename T>
SphSolver<T>
SphSolver<T>::Builder::build() const {
    validate();
    return SphSolver<T>(_fluid, _operator);
}

template <typename T>
atlas::host_shared_ptr<SphSolver<T>>
SphSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<SphSolver<T>>(_fluid, _operator);
}

}
