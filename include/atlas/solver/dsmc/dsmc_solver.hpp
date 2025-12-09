#pragma once
#include <algorithm>
#include <atlas/iterator/counting_iterator.h>
#include <atlas/math/math.h>
#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/solver/dsmc/metrics/g_ref_rms.h>
#include <atlas/transform/transform.h>
#include <cmath>
namespace atlas::solver {
template <typename T>
DsmcSolver<T>::DsmcSolver() {
    _searcher        = atlas::make_host_shared<system::SpatialHashingSearcher<T>>();
    _dsmc_data       = atlas::make_host_shared<solver::DsmcData<T>>();
    _g_ref_estimator = atlas::make_host_shared<dsmc::GRefRmsEstimator<T>>();
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcSolver<T>::solve(const system::ParticleDeviceProbe<T>& data,
                     T dt,
                     int& active) {
    (*this)(data, dt, active);
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcSolver<T>::operator()(const system::ParticleDeviceProbe<T>& data,
                          T dt,
                          int& active) {
    if (active <= 1) {
        return;
    }
    _searcher->build(data, active);
    auto neighbor_probe = _searcher->make_device_probe();
    auto n_cells        = static_cast<int>(_searcher->num_cells());
    if (n_cells <= 0) {
        return;
    }
    _dsmc_data->reset(n_cells);
    auto dsmc_probe = _dsmc_data->make_device_probe();
    compute_g_ref(neighbor_probe, dsmc_probe, data, n_cells);
    count_collisions(neighbor_probe, dsmc_probe, dt);

    auto d_n_collisions   = dsmc_probe.n_collisions;
    auto d_g_ref_per_cell = dsmc_probe.g_ref_per_cell;
    auto vel              = data.vel;

    atlas::DeviceBuffer<int> d_cell_offs(n_cells + 1);
    int* d_cell_offs_ptr = atlas::raw_pointer_cast(d_cell_offs.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(
        d_n_collisions,
        d_n_collisions + n_cells,
        d_cell_offs_ptr,
        0);

    {
        atlas::counting_iterator<int> one_begin(0);
        atlas::counting_iterator<int> one_end(1);

        atlas::parallel_for<ExecutionPolicy::device>(
            one_begin,
            one_end,
            [d_cell_offs_ptr, d_n_collisions, n_cells] ATLAS_ALL_DEVICE(int) {
                const int last_off       = d_cell_offs_ptr[n_cells - 1];
                const int last_cnt       = d_n_collisions[n_cells - 1];
                d_cell_offs_ptr[n_cells] = last_off + last_cnt;
            });
    }

    int total_collisions = 0;
    atlas::copy_device_to_host(d_cell_offs_ptr + n_cells, &total_collisions, 1);
    if (total_collisions <= 0) {
        return;
    }

    atlas::DeviceBuffer<int> d_collision_to_cell(total_collisions);
    int* d_collision_to_cell_ptr = atlas::raw_pointer_cast(d_collision_to_cell.data());

    {
        atlas::counting_iterator<int> cell_begin(0);
        atlas::counting_iterator<int> cell_end(n_cells);

        atlas::parallel_for<ExecutionPolicy::device>(
            cell_begin,
            cell_end,
            [d_n_collisions,
             d_cell_offs_ptr,
             d_collision_to_cell_ptr] ATLAS_ALL_DEVICE(int cell_id) {
                const int cnt = d_n_collisions[cell_id];
                if (cnt <= 0) {
                    return;
                }
                const int off = d_cell_offs_ptr[cell_id];
                for (int k = 0; k < cnt; ++k) {
                    d_collision_to_cell_ptr[off + k] = cell_id;
                }
            });
    }

    atlas::counting_iterator<int> coll_begin(0);
    atlas::counting_iterator<int> coll_end(total_collisions);
    const T pi = T(M_PI);

    atlas::parallel_for<ExecutionPolicy::device>(
        coll_begin,
        coll_end,
        [neighbor_probe,
         d_collision_to_cell_ptr,
         d_g_ref_per_cell,
         vel,
         pi] ATLAS_ALL_DEVICE(const int coll_id) {
            const int cell_id = d_collision_to_cell_ptr[coll_id];

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

            T Vcx = T(0.5) * (vi.x + vj.x);
            T Vcy = T(0.5) * (vi.y + vj.y);
            T Vcz = T(0.5) * (vi.z + vj.z);

            T r1        = uni(rng);
            T r2        = uni(rng);
            T cos_theta = T(1) - T(2) * r1;
            T sin_theta = std::sqrt(
                std::max(T(0), T(1) - cos_theta * cos_theta));

            T phi     = T(2) * pi * r2;
            T cos_phi = std::cos(phi);
            T sin_phi = std::sin(phi);

            T g   = g_ij;
            T grx = g * sin_theta * cos_phi;
            T gry = g * sin_theta * sin_phi;
            T grz = g * cos_theta;

            vel[i] = Vector3<T>(Vcx + T(0.5) * grx,
                                Vcy + T(0.5) * gry,
                                Vcz + T(0.5) * grz);
            vel[j] = Vector3<T>(Vcx - T(0.5) * grx,
                                Vcy - T(0.5) * gry,
                                Vcz - T(0.5) * grz);
        });
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcSolver<T>::count_collisions(const system::SpatialHashProbe<T>& neighbor_probe,
                                const DsmcDeviceProbe<T>& dsmc_probe,
                                T dt) {
    atlas::counting_iterator<int> cell_begin(0);
    atlas::counting_iterator<int> cell_end(static_cast<int>(_searcher->num_cells()));
    auto d_n_collisions = dsmc_probe.n_collisions;
    T sigma_t           = _dsmc_data->sigma_t();
    T number_weight     = _dsmc_data->number_weight();
    T cell_volume       = _dsmc_data->cell_volume();
    atlas::transform<ExecutionPolicy::device>(
        cell_begin,
        cell_end,
        d_n_collisions,
        [neighbor_probe, dsmc_probe, sigma_t, number_weight, cell_volume, dt] ATLAS_ALL_DEVICE(int cell_id) {
            const int cell_begin_idx = neighbor_probe.cell_start[cell_id];
            if (cell_begin_idx < 0) {
                return 0;
            }
            const int cell_end_idx = neighbor_probe.cell_end[cell_id];
            int n_c                = cell_end_idx - cell_begin_idx;
            if (n_c < 2) {
                return 0;
            }
            T g_ref = dsmc_probe.g_ref_per_cell[cell_id];
            if (g_ref <= T(0)) {
                return 0;
            }
            if (cell_volume <= T(0)) {
                return 0;
            }
            T n_c_t          = static_cast<T>(n_c);
            T n_pairs        = T(0.5) * n_c_t * (n_c_t - T(1));
            T scaling        = number_weight / cell_volume;
            T n_real         = sigma_t * g_ref * n_pairs * scaling * dt;
            int n_collisions = static_cast<int>(n_real + T(0.5));
            if (n_collisions < 0) {
                n_collisions = 0;
            }
            return n_collisions;
        });
}

}