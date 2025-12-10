#pragma once

#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
namespace atlas::solver::dsmc::kernel {

template <typename T>
void
HardSphereKernel<T>::operator()(const system::ParticleDeviceProbe<T>& data,
                                const DsmcDeviceProbe<T>& dsmc_probe,
                                const system::SpatialHashProbe<T>& neighbor_probe,
                                DeviceBuffer<int>& d_flattened_collision) const {
    {
        auto d_g_ref_per_cell      = dsmc_probe.g_ref_per_cell;
        auto vel                   = data.vel;
        int* d_flatten_ptr         = atlas::raw_pointer_cast(d_flattened_collision.data());
        const int total_collisions = static_cast<int>(d_flattened_collision.size());

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            total_collisions,
            [neighbor_probe,
             d_flatten_ptr,
             d_g_ref_per_cell,
             vel] ATLAS_ALL_DEVICE(const int coll_id) {
                const int cell_id = d_flatten_ptr[coll_id];

                const int first = neighbor_probe.cell_start[cell_id];
                if (first < 0) {
                    return;
                }
                const int last = neighbor_probe.cell_end[cell_id];
                int n_c        = last - first;
                if (n_c < 2) {
                    return;
                }

                const T g_ref = d_g_ref_per_cell[cell_id];
                if (g_ref <= T(0)) {
                    return;
                }

                atlas::default_random_engine<T> rng;
                atlas::uniform_real_distribution<T> uni(T(0), T(1));

                rng.seed(1234u
                         + static_cast<unsigned int>(cell_id) * 73856093u
                         + static_cast<unsigned int>(coll_id) * 19349663u);

                int i, j;
                do {
                    i = first + static_cast<int>(uni(rng) * static_cast<T>(n_c));
                    j = first + static_cast<int>(uni(rng) * static_cast<T>(n_c));
                } while (i == j);

                Vector3<T> vi = vel[i];
                Vector3<T> vj = vel[j];

                T g_ij2 = math::length_squared(vi - vj);
                if (g_ij2 <= T(0)) {
                    return;
                }

                T g_ij     = std::sqrt(g_ij2);
                T p_accept = g_ij / g_ref;
                if (p_accept <= T(0)) {
                    return;
                }

                if (p_accept < T(1)) {
                    T r = uni(rng);
                    if (r > p_accept) {
                        return;
                    }
                }

                Vector3<T> v_c = (vi + vj) * T(0.5);
                T r1           = uni(rng);
                T r2           = uni(rng);
                T cos_theta    = T(1) - T(2) * r1;
                T sin_theta    = std::sqrt(std::max(T(0), T(1) - cos_theta * cos_theta));

                T phi     = T(2) * pi * r2;
                T cos_phi = std::cos(phi);
                T sin_phi = std::sin(phi);

                T g            = g_ij;
                Vector3<T> g_r = Vector3<T> { g * sin_theta * cos_phi,
                                              g * sin_theta * sin_phi,
                                              g * cos_theta };
                vel[i]         = v_c + T(0.5) * g_r;
                vel[j]         = v_c - T(0.5) * g_r;
            });
    }
}

template <typename T>
void
HardSphereKernel<T>::collide(const system::ParticleDeviceProbe<T>& data,
                             const DsmcDeviceProbe<T>& dsmc_probe,
                             const system::SpatialHashProbe<T>& neighbor_probe,
                             DeviceBuffer<int>& d_flattened_collision) const {
    this->operator()(data, dsmc_probe, neighbor_probe, d_flattened_collision);
}

}