#pragma once
#include <atlas/iterator/counting_iterator.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/transform/transform.h>
#include <atlas/solver/dsmc/metrics/g_ref_rms.h>
#include <atlas/memory/copy.h>
#include <cmath>
#include <algorithm>
namespace atlas::solver {
template <typename T>
DsmcSolver<T>::DsmcSolver() {
_searcher  = atlas::make_host_shared<system::SpatialHashingSearcher<T>>();
_dsmc_data = atlas::make_host_shared<solver::DsmcData<T>>();
_g_ref_estimator        = atlas::make_host_shared<dsmc::GRefRmsEstimator<T>>();
_dsmc_work_distribution = DsmcWorkDistribution::PerCollision;
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
if (_dsmc_work_distribution == DsmcWorkDistribution::PerCell) {
collide_per_cell(neighbor_probe, dsmc_probe, data, dt);
return;
}
if (_dsmc_work_distribution == DsmcWorkDistribution::PerCollision) {
prepare_per_collision_prefix_sum(dsmc_probe, n_cells);
collide_per_collision(neighbor_probe, dsmc_probe, data, dt);
return;
}
}
template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
DsmcSolver<T>::count_collisions(const system::SpatialHashProbe<T>& neighbor_probe,
const DsmcDeviceProbe<T>& dsmc_probe,
T dt) {
atlas::counting_iterator<int> cell_begin(0);
atlas::counting_iterator<int> cell_end(static_cast<int>(_searcher->num_cells()));
auto d_n_collisions = dsmc_probe.n_collisions;
T sigma_t       = _dsmc_data->sigma_t();
T number_weight = _dsmc_data->number_weight();
T cell_volume   = _dsmc_data->cell_volume();
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
T n_c_t   = static_cast<T>(n_c);
T n_pairs = T(0.5) * n_c_t * (n_c_t - T(1));
T scaling = number_weight / cell_volume;
T n_real = sigma_t * g_ref * n_pairs * scaling * dt;
int n_collisions = static_cast<int>(n_real + T(0.5));
if (n_collisions < 0) {
n_collisions = 0;
}
return n_collisions;
});
}
template <typename T>
void
DsmcSolver<T>::prepare_per_collision_prefix_sum(const DsmcDeviceProbe<T>& dsmc_probe,
int n_cells) {
auto d_n_collisions   = dsmc_probe.n_collisions;
auto d_collision_offs = dsmc_probe.collision_offset;
atlas::exclusive_scan<ExecutionPolicy::device>(
d_n_collisions,
d_n_collisions + n_cells,
d_collision_offs,
0);
int last_offset      = 0;
int last_n_collision = 0;
atlas::copy_device_to_host(d_collision_offs + (n_cells - 1), &last_offset, 1);
atlas::copy_device_to_host(d_n_collisions + (n_cells - 1), &last_n_collision, 1);
int total = last_offset + last_n_collision;
_dsmc_data->set_total_collisions(total);
}
template <typename T>
void
DsmcSolver<T>::collide_per_cell(const system::SpatialHashProbe<T>& neighbor_probe,
const DsmcDeviceProbe<T>& dsmc_probe,
const system::ParticleDeviceProbe<T>& data,
T ) {
const int n_cells = static_cast<int>(_searcher->num_cells());
atlas::counting_iterator<int> cell_begin(0);
atlas::counting_iterator<int> cell_end(n_cells);
auto d_n_collisions   = dsmc_probe.n_collisions;
auto d_g_ref_per_cell = dsmc_probe.g_ref_per_cell;
auto vel              = data.vel;
atlas::parallel_for<ExecutionPolicy::device>(
cell_begin,
cell_end,
[neighbor_probe, d_n_collisions, d_g_ref_per_cell, vel] ATLAS_ALL_DEVICE(int cell_id) {
const int first = neighbor_probe.cell_start[cell_id];
if (first < 0) {
return;
}
const int last = neighbor_probe.cell_end[cell_id];
int n_c        = last - first;
if (n_c < 2) {
return;
}
int n_collisions = d_n_collisions[cell_id];
if (n_collisions <= 0) {
return;
}
const T g_ref = d_g_ref_per_cell[cell_id];
if (g_ref <= T(0)) {
return;
}
atlas::default_random_engine<T> rng;
atlas::uniform_real_distribution<T> uni(T(0), T(1));
rng.seed(1234u + static_cast<unsigned int>(cell_id));
const T pi = T(3.14159265358979323846);
for (int c = 0; c < n_collisions; ++c) {
int i, j;
do {
i = first + static_cast<int>(uni(rng) * static_cast<T>(n_c));
j = first + static_cast<int>(uni(rng) * static_cast<T>(n_c));
} while (i == j);
Vector3<T> vi = vel[i];
Vector3<T> vj = vel[j];
T g_ij2 = math::length_squared(vi - vj);
if (g_ij2 <= T(0)) {
continue;
}
T g_ij = std::sqrt(g_ij2);
T p_accept = g_ij / g_ref;
if (p_accept <= T(0)) {
continue;
}
if (p_accept < T(1)) {
T r = uni(rng);
if (r > p_accept) {
continue;
}
}
T Vcx = T(0.5) * (vi.x + vj.x);
T Vcy = T(0.5) * (vi.y + vj.y);
T Vcz = T(0.5) * (vi.z + vj.z);
T r1 = uni(rng);
T r2 = uni(rng);
T cos_theta = T(1) - T(2) * r1;
T sin_theta = std::sqrt(
std::max(T(0), T(1) - cos_theta * cos_theta));
T phi     = T(2) * pi * r2;
T cos_phi = std::cos(phi);
T sin_phi = std::sin(phi);
T g = g_ij;
T grx = g * sin_theta * cos_phi;
T gry = g * sin_theta * sin_phi;
T grz = g * cos_theta;
vel[i] = Vector3<T>(Vcx + T(0.5) * grx,
Vcy + T(0.5) * gry,
Vcz + T(0.5) * grz);
vel[j] = Vector3<T>(Vcx - T(0.5) * grx,
Vcy - T(0.5) * gry,
Vcz - T(0.5) * grz);
}
});
}
template <typename T>
void
DsmcSolver<T>::collide_per_collision(const system::SpatialHashProbe<T>& neighbor_probe,
const DsmcDeviceProbe<T>& dsmc_probe,
const system::ParticleDeviceProbe<T>& data,
T ) {
const int n_cells          = static_cast<int>(_searcher->num_cells());
const int total_collisions = _dsmc_data->total_collisions();
if (total_collisions <= 0) {
return;
}
auto d_n_collisions   = dsmc_probe.n_collisions;
auto d_collision_offs = dsmc_probe.collision_offset;
auto d_g_ref_per_cell = dsmc_probe.g_ref_per_cell;
auto vel              = data.vel;
atlas::counting_iterator<int> coll_begin(0);
atlas::counting_iterator<int> coll_end(total_collisions);
const T pi = T(3.14159265358979323846);
atlas::parallel_for<ExecutionPolicy::device>(
coll_begin,
coll_end,
[neighbor_probe,
d_n_collisions,
d_collision_offs,
d_g_ref_per_cell,
vel,
n_cells,
pi] ATLAS_ALL_DEVICE(int coll_id) {
int left    = 0;
int right   = n_cells;
int cell_id = -1;
while (left < right) {
int mid  = (left + right) / 2;
int off  = d_collision_offs[mid];
int cnt  = d_n_collisions[mid];
int offN = off + cnt;
if (coll_id < off) {
right = mid;
} else if (coll_id >= offN) {
left = mid + 1;
} else {
cell_id = mid;
break;
}
}
if (cell_id < 0) {
return;
}
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
T g_ij = std::sqrt(g_ij2);
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
T r1 = uni(rng);
T r2 = uni(rng);
T cos_theta = T(1) - T(2) * r1;
T sin_theta = std::sqrt(
std::max(T(0), T(1) - cos_theta * cos_theta));
T phi     = T(2) * pi * r2;
T cos_phi = std::cos(phi);
T sin_phi = std::sin(phi);
T g = g_ij;
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
}