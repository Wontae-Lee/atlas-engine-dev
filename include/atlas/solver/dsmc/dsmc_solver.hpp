#pragma once
#include <atlas/functor/device_pair_indexer.h>
#include <atlas/iterator/counting_iterator.h>
#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/solver/dsmc/kernel/hard_sphere.h>
#include <atlas/solver/dsmc/metrics/g_ref_rms.h>

namespace atlas::solver {
template <typename T>
DsmcSolver<T>::DsmcSolver() {
    _searcher        = atlas::make_host_shared<system::SpatialHashingSearcher<T>>();
    _dsmc_data       = atlas::make_host_shared<solver::DsmcData<T>>();
    _g_ref_estimator = atlas::make_host_shared<dsmc::GRefRmsEstimator<T>>();
    _collide_kernel  = atlas::make_host_shared<dsmc::HardSphereKernel<T>>();
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
    auto n_cells        = static_cast<int>(_searcher->n_cells());
    if (n_cells <= 0) {
        return;
    }

    _dsmc_data->reset(n_cells);
    auto dsmc_probe = _dsmc_data->make_device_probe();
    count_species(neighbor_probe, dsmc_probe, data);
    compute_g_ref(neighbor_probe, dsmc_probe, data);
    count_collisions(neighbor_probe, dsmc_probe, dt);

    atlas::DeviceBuffer<int> d_flattened_collision;
    flatten_collision(dsmc_probe, d_flattened_collision);
    if (d_flattened_collision.size() <= 0) {
        return;
    }
    collide_particles(data,
                      dsmc_probe,
                      neighbor_probe,
                      d_flattened_collision);
}

template <typename T>
void
DsmcSolver<T>::collide_particles(const system::ParticleDeviceProbe<T>& data,
                                 const DsmcDeviceProbe<T>& dsmc_probe,
                                 const system::SpatialHashProbe<T>& neighbor_probe,
                                 DeviceBuffer<int>& d_flattened_collision) {
    if (_collide_kernel != nullptr) {
        (*_collide_kernel)(data,
                           dsmc_probe,
                           neighbor_probe,
                           d_flattened_collision);
    }
}

template <typename T>
void
DsmcSolver<T>::flatten_collision(const DsmcDeviceProbe<T>& dsmc_probe,
                                 DeviceBuffer<int>& d_flattened_collision) {

    auto d_n_collisions = dsmc_probe.n_collisions;
    auto n_cells        = static_cast<int>(_searcher->n_cells());

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
    d_flattened_collision.resize(total_collisions);

    int* d_flatten_ptr = atlas::raw_pointer_cast(d_flattened_collision.data());
    {
        atlas::counting_iterator<int> cell_begin(0);
        atlas::counting_iterator<int> cell_end(n_cells);

        atlas::parallel_for<ExecutionPolicy::device>(
            cell_begin,
            cell_end,
            [d_n_collisions,
             d_cell_offs_ptr,
             d_flatten_ptr] ATLAS_ALL_DEVICE(int cell_id) {
                const int cnt = d_n_collisions[cell_id];
                if (cnt <= 0) {
                    return;
                }
                const int off = d_cell_offs_ptr[cell_id];
                for (int k = 0; k < cnt; ++k) {
                    d_flatten_ptr[off + k] = cell_id;
                }
            });
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcSolver<T>::count_collisions(const system::SpatialHashProbe<T>& neighbor_probe,
                                const DsmcDeviceProbe<T>& dsmc_probe,
                                T dt) {
    const int n_cells     = static_cast<int>(_searcher->n_cells());
    const int n_species   = _dsmc_data->n_species();
    const int n_pairs     = _dsmc_data->n_pairs();
    const int total_pairs = n_cells * n_pairs;

    T cell_volume    = _dsmc_data->cell_volume();
    T* sigma_pairs   = _dsmc_data->sigma_t_pairs_ptr();
    T* number_weight = _dsmc_data->number_weight_ptr();

    T* d_g_ref         = dsmc_probe.g_ref_per_cell;
    int* d_n_coll      = dsmc_probe.n_collisions;
    int* d_n_c_species = dsmc_probe.n_particles_per_cell_species;

    if (cell_volume <= T(0)) {
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            total_pairs,
            [d_n_coll] ATLAS_ALL_DEVICE(const int idx) {
                d_n_coll[idx] = 0;
            });
        return;
    }

    auto pair_index_device = DevicePairIndexer<int>();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_cells,
        [neighbor_probe,
         d_g_ref,
         d_n_coll,
         d_n_c_species,
         sigma_pairs,
         number_weight,
         n_species,
         n_pairs,
         cell_volume,
         dt,
         pair_index_device] ATLAS_ALL_DEVICE(int cell_id) {
            if (!d_g_ref || !d_n_coll || !d_n_c_species) {
                return;
            }

            const int cell_begin_idx = neighbor_probe.cell_start[cell_id];

            if (cell_begin_idx < 0) {
                for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
                    d_n_coll[cell_id * n_pairs + pair_idx] = 0;
                }
                return;
            }

            const int cell_end_idx = neighbor_probe.cell_end[cell_id];
            if (const int n_c = cell_end_idx - cell_begin_idx; n_c < 2) {
                for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
                    d_n_coll[cell_id * n_pairs + pair_idx] = 0;
                }
                return;
            }

            // single-species fast path
            if (n_species == 1 && n_pairs == 1) {
                constexpr int pair_idx = 0;
                const int offset       = cell_id * n_pairs + pair_idx;
                const int n_c_s        = d_n_c_species[cell_id * n_species + 0];

                if (n_c_s < 2) {
                    d_n_coll[offset] = 0;
                    return;
                }

                const T g_ref = d_g_ref[offset];
                if (g_ref <= T(0)) {
                    d_n_coll[offset] = 0;
                    return;
                }

                const T sigma  = sigma_pairs[pair_idx];
                const T weight = number_weight[0];
                if (sigma <= T(0) || weight <= T(0)) {
                    d_n_coll[offset] = 0;
                    return;
                }

                const T n_c_t     = static_cast<T>(n_c_s);
                const T n_pairs_s = T(0.5) * n_c_t * (n_c_t - T(1));
                const T scaling   = weight / cell_volume;

                const T n_real = sigma * g_ref * n_pairs_s * scaling * dt;
                int n_coll     = static_cast<int>(n_real + T(0.5));
                if (n_coll < 0) n_coll = 0;

                d_n_coll[offset] = n_coll;
                return;
            }

            // multi-species path
            for (int s = 0; s < n_species; ++s) {
                const int ns = d_n_c_species[cell_id * n_species + s];

                for (int r = s; r < n_species; ++r) {
                    const int nr = d_n_c_species[cell_id * n_species + r];

                    const int pair_idx = pair_index_device.pair_index(s, r, n_species);

                    if (pair_idx < 0 || pair_idx >= n_pairs) {
                        return;
                    }

                    const int offset = cell_id * n_pairs + pair_idx;

                    if ((s == r && ns < 2) || (s != r && (ns == 0 || nr == 0))) {
                        d_n_coll[offset] = 0;
                        continue;
                    }

                    const T g_ref = d_g_ref[offset];
                    if (g_ref <= T(0)) {
                        d_n_coll[offset] = 0;
                        continue;
                    }

                    const T sigma_sr = sigma_pairs[pair_idx];
                    if (sigma_sr <= T(0)) {
                        d_n_coll[offset] = 0;
                        continue;
                    }

                    T n_pairs_sr;
                    if (s == r) {
                        const T ns_t = static_cast<T>(ns);
                        n_pairs_sr   = T(0.5) * ns_t * (ns_t - T(1));
                    } else {
                        n_pairs_sr = static_cast<T>(ns) * static_cast<T>(nr);
                    }

                    const T w_s     = number_weight[s];
                    const T scaling = w_s / cell_volume;

                    const T n_real = sigma_sr * g_ref * n_pairs_sr * scaling * dt;
                    int n_coll     = static_cast<int>(n_real + T(0.5));
                    if (n_coll < 0) n_coll = 0;

                    d_n_coll[offset] = n_coll;
                }
            }
        });
}

template <typename T>
void
DsmcSolver<T>::compute_g_ref(const system::SpatialHashProbe<T>& neighbor_probe,
                             const DsmcDeviceProbe<T>& dsmc_probe,
                             const system::ParticleDeviceProbe<T>& data) {
    const int n_cells   = static_cast<int>(_searcher->n_cells());
    const int n_species = _dsmc_data->n_species();
    const int n_pairs   = _dsmc_data->n_pairs();

    auto d_g_ref = dsmc_probe.g_ref_per_cell; // size = n_cells * n_pairs
    auto gref_op = _g_ref_estimator->make_device_operator();
    ATLAS_ASSERT(d_g_ref != nullptr);

    // single-species fast path (n_pairs == 1)
    if (n_species == 1 && n_pairs == 1) {
        atlas::parallel_for<ExecutionPolicy::device>(

            0,
            n_cells,
            [neighbor_probe,
             data,
             gref_op,
             d_g_ref] ATLAS_ALL_DEVICE(int cell_id) {
                const int begin = neighbor_probe.cell_start[cell_id];
                if (begin < 0) {
                    d_g_ref[cell_id] = T(0);
                    return;
                }

                // g_ref는 셀 단위 스칼라
                const T g_ref    = gref_op(cell_id, neighbor_probe, data);
                d_g_ref[cell_id] = (g_ref > T(0)) ? g_ref : T(0);
            });
        return;
    }

    // multi-species path
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_cells,
        [neighbor_probe,
         data,
         gref_op,
         d_g_ref,
         n_pairs] ATLAS_ALL_DEVICE(int cell_id) {
            const int begin = neighbor_probe.cell_start[cell_id];
            if (begin < 0) {
                const int base = cell_id * n_pairs;
                for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
                    d_g_ref[base + pair_idx] = T(0);
                }
                return;
            }

            const T g_ref = gref_op(cell_id, neighbor_probe, data);
            const T g_val = (g_ref > T(0)) ? g_ref : T(0);

            const int base = cell_id * n_pairs;
            for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
                d_g_ref[base + pair_idx] = g_val;
            }
        });
}

template <typename T>
void
DsmcSolver<T>::count_species(const system::SpatialHashProbe<T>& neighbor_probe,
                             const DsmcDeviceProbe<T>& dsmc_probe,
                             const system::ParticleDeviceProbe<T>& data) {
    const int n_cells   = static_cast<int>(_searcher->n_cells());
    const int n_species = _dsmc_data->n_species();
    auto d_species      = data.species;

    auto d_n_c_spec = dsmc_probe.n_particles_per_cell_species;
    ATLAS_ASSERT(d_n_c_spec != nullptr);

    if (n_species == 1) {
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            n_cells,
            [neighbor_probe,
             d_n_c_spec] ATLAS_ALL_DEVICE(int cell_id) {
                const int cell_begin = neighbor_probe.cell_start[cell_id];
                if (cell_begin < 0) {
                    d_n_c_spec[cell_id] = 0;
                    return;
                }
                const int cell_end  = neighbor_probe.cell_end[cell_id];
                d_n_c_spec[cell_id] = cell_end - cell_begin;
            });
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_cells,
        [neighbor_probe,
         d_species,
         d_n_c_spec,
         n_species] ATLAS_ALL_DEVICE(int cell_id) {
            const int cell_begin = neighbor_probe.cell_start[cell_id];
            const int base       = cell_id * n_species;

            if (cell_begin < 0) {
                for (int s = 0; s < n_species; ++s) {
                    d_n_c_spec[base + s] = 0;
                }
                return;
            }

            const int cell_end = neighbor_probe.cell_end[cell_id];

            for (int s = 0; s < n_species; ++s) {
                d_n_c_spec[base + s] = 0;
            }

            for (int idx = cell_begin; idx < cell_end; ++idx) {
                const int p_id = neighbor_probe.indices[idx];
                const int s    = d_species[p_id];
                if (s >= 0 && s < n_species) {
                    ++d_n_c_spec[base + s];
                }
            }
        });
}

}