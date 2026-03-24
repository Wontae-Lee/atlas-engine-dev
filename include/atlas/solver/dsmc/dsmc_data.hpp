// #pragma once
// #include <algorithm>
//
// namespace atlas::solver {
// template <typename T>
// DsmcData<T>::DsmcData()
//     : d_g_ref_per_cell()
//       , d_n_collisions()
//       , d_n_particles_per_cell_species()
//       , d_sigma_t_pairs()
//       , d_number_weight_species()
//       , _cell_volume(T(1))
//       , _n_species(1)
//       , _n_pairs(1)
//       , _n_cells(0) {
//     set_n_species(1);
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE void
// DsmcData<T>::set_n_species(const int n_species) {
//     _n_species = std::max(1, n_species);
//     _n_pairs   = _n_species * (_n_species + 1) / 2;
//
//     d_sigma_t_pairs.resize(_n_pairs, T(1));
//     d_number_weight_species.resize(_n_species, T(1));
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE int
// DsmcData<T>::pair_index(int s, int r) const {
//     if (r < s) {
//         const int t = s;
//         s           = r;
//         r           = t;
//     }
//     return s * _n_species - (s * (s - 1)) / 2 + (r - s);
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE int
// DsmcData<T>::n_species() const {
//     return _n_species;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE int
// DsmcData<T>::n_pairs() const {
//     return _n_pairs;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE void
// DsmcData<T>::set_sigma_t(const int s, const int r, T value) {
//     d_sigma_t_pairs[pair_index(s, r)] = value;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE T
// DsmcData<T>::sigma_t(int idx) const {
//     return d_sigma_t_pairs[idx];
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE void
// DsmcData<T>::set_number_weight(int s, T value) {
//     d_number_weight_species[s] = value;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE T
// DsmcData<T>::number_weight(int s) const {
//     return d_number_weight_species[s];
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE void
// DsmcData<T>::set_cell_volume(T v) {
//     _cell_volume = v;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE T
// DsmcData<T>::cell_volume() const {
//     return _cell_volume;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE T*
// DsmcData<T>::sigma_t_pairs_ptr() {
//     return atlas::raw_pointer_cast(d_sigma_t_pairs.data());
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE T*
// DsmcData<T>::number_weight_ptr() {
//     return atlas::raw_pointer_cast(d_number_weight_species.data());
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE DsmcDeviceProbe<T>
//
// DsmcData<T>::make_device_probe() noexcept {
//     DsmcDeviceProbe<T> probe{};
//     probe.g_ref_per_cell               = atlas::raw_pointer_cast(d_g_ref_per_cell.data());
//     probe.n_collisions                 = atlas::raw_pointer_cast(d_n_collisions.data());
//     probe.n_particles_per_cell_species = atlas::raw_pointer_cast(d_n_particles_per_cell_species.data());
//     return probe;
// }
//
// template <typename T>
// ATLAS_HOST ATLAS_FORCE_INLINE void
// DsmcData<T>::reset(const int n_cells) {
//     _n_cells = n_cells;
//
//     const int total_cell_pair    = _n_cells * _n_pairs;
//     const int total_cell_species = _n_cells * _n_species;
//
//     d_g_ref_per_cell.resize(total_cell_pair, T(0));
//     d_n_collisions.resize(total_cell_pair, 0);
//     d_n_particles_per_cell_species.resize(total_cell_species, 0);
// }
// }